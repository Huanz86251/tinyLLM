"""Verified IFEval OPD for tinyLLM with explicit 60/20/20 loss balancing."""
from __future__ import annotations

import argparse
import collections
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT / "runtime" / "minicpm_transformers_449"
if not (COMPAT / "transformers" / "__init__.py").is_file():
    raise FileNotFoundError(COMPAT)
sys.path.insert(0, str(COMPAT))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from project_paths import path
from train import GRPO as g
from train.opd_gsm8k_stage import chunked_jsd, is_loop, learning_rate, save_json
from train.opd_hybrid_v3 import (
    FP32MasterAdamW, assistant_only_ce, evaluate_general_loss, file_sha256,
    load_general_pools, sequence_logits,
)


def read_jsonl(file: Path) -> list[dict]:
    with file.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_verifiers(file: Path) -> dict:
    spec = importlib.util.spec_from_file_location("tinyllm_ifeval_functions", file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verifier file: {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IF_FUNCTIONS_MAP


def verify_response(row: dict, text: str, verifiers: dict) -> bool:
    try:
        return bool(verifiers[row["verifier"]](text, **row["verifier_args"]))
    except Exception:
        return False


def validate_config(cfg: dict) -> None:
    composition = cfg["composition"]
    counts = (
        int(composition["student_on_policy_ifeval"]),
        int(composition["teacher_verified_ifeval"]),
        int(composition["general_sft_per_update"]),
    )
    if counts != (10, 3, 3) or sum(counts) != composition["effective_global_batch"]:
        raise ValueError("IFEval composition must be 10 student + 3 teacher + 3 replay = 16")
    shares = composition["target_objective_share"]
    if not math.isclose(sum(float(x) for x in shares.values()), 1.0):
        raise ValueError("target objective shares must sum to one")
    if any(float(v) <= 0 for v in shares.values()):
        raise ValueError("objective shares must be positive")
    if cfg["training"]["physical_microbatch"] != 1:
        raise ValueError("physical_microbatch must remain one")
    language_cycle = composition["general_language_cycle"]
    if not language_cycle or any(x["chinese"] + x["english"] != 3 for x in language_cycle):
        raise ValueError("each replay cycle item must contain three rows")


def load_data(cfg: dict) -> tuple[list[dict], list[dict], dict, dict]:
    data_cfg = cfg["dataset"]["ifeval"]
    manifest_file = path(data_cfg["manifest_file"])
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    train_file = path(data_cfg["train_file"])
    dev_file = path(data_cfg["dev_file"])
    if file_sha256(train_file) != manifest["files"]["train"]["sha256"]:
        raise RuntimeError("IFEval train checksum mismatch")
    if file_sha256(dev_file) != manifest["files"]["dev"]["sha256"]:
        raise RuntimeError("IFEval dev checksum mismatch")
    verifier_file = path(data_cfg["verifier_file"])
    if file_sha256(verifier_file) != manifest["verifier_sha256"]:
        raise RuntimeError("IFEval verifier checksum mismatch")
    return read_jsonl(train_file), read_jsonl(dev_file), manifest, load_verifiers(verifier_file)


def load_teacher_pool(cfg: dict, train_rows: list[dict], verifiers: dict) -> tuple[list[dict], dict]:
    directory = path(cfg["teacher_pool"]["output_dir"])
    manifest_file = directory / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"teacher pool is not complete: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        raise RuntimeError("teacher pool manifest is incomplete")
    rows_file = Path(manifest["rows_file"])
    if file_sha256(rows_file) != manifest["rows_sha256"]:
        raise RuntimeError("teacher pool checksum mismatch")
    all_rows = read_jsonl(rows_file)
    if len(all_rows) != len(train_rows) or len(all_rows) != manifest["total_scanned"]:
        raise RuntimeError("teacher pool is not an exhaustive training scan")
    qualified = [row for row in all_rows if row["qualified"]]
    if len(qualified) != manifest["qualified"]:
        raise RuntimeError("teacher pool qualified count mismatch")
    for row in qualified:
        if not verify_response(row, row["accepted_text"], verifiers):
            raise RuntimeError(f"cached teacher answer failed revalidation: {row['source_index']}")
    return qualified, manifest


@torch.inference_mode()
def generate_student_batch(model, tokenizer, prompts: list[str], cfg: dict) -> list[dict]:
    sampled = g.sample_for_grpo_prompt_batch(
        model, tokenizer, prompts, None,
        max_new_tokens=int(cfg["student_rollout_max_new_tokens"]),
        temperature=float(cfg["student_temperature"]),
        top_p=float(cfg["student_top_p"]),
        repetition_penalty=float(cfg["student_repetition_penalty"]),
        no_repeat_ngram_size=int(cfg["student_no_repeat_ngram_size"]),
    )
    return [
        {"text": text, "token_ids": list(token_ids), "generation_batch_size": len(prompts)}
        for text, token_ids in zip(sampled["texts"], sampled["token_ids"])
    ]


def on_policy_loss(student, teacher, tokenizer, row: dict, sample: dict,
                   training: dict, verifiers: dict) -> tuple[torch.Tensor, dict]:
    completion = sample["token_ids"]
    text = sample["text"]
    if not completion:
        raise RuntimeError("student generated an empty completion")
    rendered = g.render_prompt_with_tokenizer(tokenizer, row["prompt"], None)
    context = tokenizer.encode(rendered, add_special_tokens=False)
    with torch.no_grad():
        teacher_out, teacher_logits, teacher_full, teacher_mask = sequence_logits(
            teacher, context, completion)
        teacher_logits = teacher_logits.detach()
    student.train()
    student_out, student_logits, student_full, student_mask = sequence_logits(
        student, context, completion)
    jsd = chunked_jsd(
        student_logits, teacher_logits, float(training["jsd_beta"]),
        float(training["distillation_temperature"]), int(training["jsd_chunk_tokens"]),
    )
    passed = verify_response(row, text, verifiers)
    if passed:
        targets = torch.tensor(completion, dtype=torch.long, device=student_logits.device)
        self_ce = F.cross_entropy(student_logits[0].float(), targets, reduction="mean")
    else:
        targets = None
        self_ce = student_logits.sum() * 0.0
    self_weight = float(training["student_verified_self_ce_weight"])
    loss = jsd + self_weight * self_ce
    meta = {
        "source_index": row["source_index"], "tokens": len(completion),
        "generation_batch_size": sample["generation_batch_size"],
        "verifier_passed": bool(passed), "loop": bool(is_loop(text)),
        "hit_token_cap": len(completion) >= int(training["student_rollout_max_new_tokens"]),
        "jsd": float(jsd.detach()), "verified_self_ce": float(self_ce.detach()),
        "text_preview": text[:200],
    }
    del teacher_out, teacher_logits, teacher_full, teacher_mask
    del student_out, student_logits, student_full, student_mask, jsd, self_ce
    if targets is not None:
        del targets
    return loss, meta


def teacher_anchor_loss(student, teacher, tokenizer, row: dict,
                        training: dict, verifiers: dict) -> tuple[torch.Tensor, dict]:
    text = row["accepted_text"]
    completion = list(row["accepted_token_ids"])
    if not completion or not verify_response(row, text, verifiers) or is_loop(text):
        raise RuntimeError(f"invalid teacher anchor {row['source_index']}")
    rendered = g.render_prompt_with_tokenizer(tokenizer, row["prompt"], None)
    context = tokenizer.encode(rendered, add_special_tokens=False)
    with torch.no_grad():
        teacher_out, teacher_logits, teacher_full, teacher_mask = sequence_logits(
            teacher, context, completion)
        teacher_logits = teacher_logits.detach()
    student.train()
    student_out, student_logits, student_full, student_mask = sequence_logits(
        student, context, completion)
    jsd = chunked_jsd(
        student_logits, teacher_logits, float(training["jsd_beta"]),
        float(training["distillation_temperature"]), int(training["jsd_chunk_tokens"]),
    )
    targets = torch.tensor(completion, dtype=torch.long, device=student_logits.device)
    ce = F.cross_entropy(student_logits[0].float(), targets, reduction="mean")
    loss = float(training["teacher_jsd_weight"]) * jsd + float(training["teacher_ce_weight"]) * ce
    meta = {"source_index": row["source_index"], "tokens": len(completion),
            "verifier_passed": True, "jsd": float(jsd.detach()), "ce": float(ce.detach()),
            "text_preview": text[:200]}
    del teacher_out, teacher_logits, teacher_full, teacher_mask
    del student_out, student_logits, student_full, student_mask, targets, jsd, ce
    return loss, meta


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    rendered = [g.render_prompt_with_tokenizer(tokenizer, prompt, None) for prompt in prompts]
    encoded = [torch.tensor(tokenizer.encode(x, add_special_tokens=False), dtype=torch.long)
               for x in rendered]
    batch = len(encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(x.numel() for x in encoded)
    input_ids = torch.full((batch, width), int(pad_id), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for index, ids in enumerate(encoded):
        input_ids[index, -ids.numel():] = ids.to("cuda")
        attention[index, -ids.numel():] = 1
    output = model(input_ids=input_ids, attention_mask=attention, labels=None,
                   use_cache=True, past_states=None)
    past_states = output["past_states"]
    logits = output["logits"][:, -1, :]
    generated = [[] for _ in range(batch)]
    finished = [False] * batch
    stop_ids = {int(x) for x in (tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>")) if x is not None and int(x) >= 0}
    for _ in range(max_new_tokens):
        next_ids = torch.argmax(logits.float(), dim=-1, keepdim=True)
        for index in range(batch):
            if finished[index]:
                next_ids[index, 0] = int(pad_id)
                continue
            token = int(next_ids[index, 0])
            if token in stop_ids:
                finished[index] = True
            else:
                generated[index].append(token)
        if all(finished):
            break
        output = model(input_ids=next_ids, attention_mask=torch.ones_like(next_ids),
                       labels=None, use_cache=True, past_states=past_states)
        past_states = output["past_states"]
        logits = output["logits"][:, -1, :]
    return [tokenizer.decode(ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False).strip() for ids in generated]


def evaluate_ifeval(model, tokenizer, rows: list[dict], verifiers: dict,
                    batch_size: int, max_new_tokens: int) -> dict:
    model.eval()
    passed = 0
    by_type = collections.defaultdict(lambda: [0, 0])
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        texts = greedy_generate(model, tokenizer, [row["prompt"] for row in batch], max_new_tokens)
        for row, text in zip(batch, texts):
            ok = verify_response(row, text, verifiers)
            passed += int(ok)
            by_type[row["constraint_type"]][0] += int(ok)
            by_type[row["constraint_type"]][1] += 1
    return {"correct": passed, "total": len(rows), "accuracy": passed / len(rows),
            "by_type": {key: {"correct": value[0], "total": value[1],
                        "accuracy": value[0] / value[1]} for key, value in sorted(by_type.items())}}


def save_checkpoint(file: Path, student, adapter_name: str, optimizer, state: dict,
                    report: dict) -> None:
    payload = {"schema_version": 1, "adapter_name": adapter_name,
               "adapter": student.get_lora_state_dict(adapter_name),
               "optimizer": optimizer.state_dict(), "state": state, "report": report,
               "python_random_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
               "cuda_rng_state": torch.cuda.get_rng_state_all()}
    temporary = Path(str(file) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    validate_config(cfg)
    train_rows, dev_rows, data_manifest, verifiers = load_data(cfg)
    pools, general_manifest = load_general_pools(cfg)
    qualified, teacher_manifest = load_teacher_pool(cfg, train_rows, verifiers)
    training = cfg["training"]
    composition = cfg["composition"]
    student_per_update = int(composition["student_on_policy_ifeval"])
    updates_per_epoch = math.ceil(len(qualified) / student_per_update)
    total_updates = updates_per_epoch * int(training["epochs"])
    run_limit = min(total_updates, args.max_updates) if args.max_updates else total_updates

    seed = int(training["seed"])
    index_by_source = {row["source_index"]: row for row in train_rows}
    qualified_data = [index_by_source[row["source_index"]] for row in qualified]
    qualified_cache = {row["source_index"]: row for row in qualified}
    student_schedule = []
    for epoch in range(int(training["epochs"])):
        epoch_rows = list(qualified_data)
        random.Random(seed + epoch).shuffle(epoch_rows)
        while len(epoch_rows) % student_per_update:
            epoch_rows.append(epoch_rows[len(epoch_rows) % len(qualified_data)])
        student_schedule.extend(epoch_rows)
    teacher_schedule = []
    needed_teacher = total_updates * int(composition["teacher_verified_ifeval"])
    cycle = list(qualified)
    round_no = 0
    while len(teacher_schedule) < needed_teacher:
        current = list(cycle)
        random.Random(seed + 1000 + round_no).shuffle(current)
        teacher_schedule.extend(current)
        round_no += 1
    teacher_schedule = teacher_schedule[:needed_teacher]

    ready = {"status": "ready", "qualified_prompts": len(qualified),
             "teacher_pass_rate": teacher_manifest["pass_rate"],
             "updates_per_epoch": updates_per_epoch, "epochs": training["epochs"],
             "total_updates": total_updates, "run_limit": run_limit,
             "logical_batch": 16, "physical_microbatch": 1,
             "target_objective_share": composition["target_objective_share"]}
    if args.dry_run:
        print(json.dumps(ready, ensure_ascii=False, indent=2))
        return
    if transformers.__version__ != "4.49.0":
        raise RuntimeError(f"expected Transformers 4.49.0, got {transformers.__version__}")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    minimum = float(cfg["constraints"]["minimum_free_vram_gb_before_start"])
    if free_bytes < minimum * 2**30:
        raise RuntimeError(f"need {minimum:.0f} GB free VRAM, found {free_bytes/2**30:.2f} GB")

    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = path("runs/training") / cfg["run_name"] / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report = {"status": "starting", "config": cfg, "run_dir": str(run_dir),
              "data_manifest": data_manifest, "teacher_manifest": teacher_manifest,
              "general_manifest": general_manifest, "history": [], "evaluations": [],
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    state = {"update": 0, "student_cursor": 0, "teacher_cursor": 0,
             "general_cursors": {"chinese": 0, "english": 0},
             "loss_ema": {}, "student_passed": 0, "student_samples": 0,
             "student_loops": 0, "student_truncated": 0}
    save_json(report_path, report)
    try:
        teacher_path = path(cfg["teacher"]["path"])
        tokenizer = AutoTokenizer.from_pretrained(
            str(teacher_path), trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        student_cfg = cfg["student"]
        student = g.load_tinyllm_from_ckpt(str(path(student_cfg["path"])), tokenizer, strict=True)
        if hasattr(student, "cfg"):
            student.cfg.use_checkpoint = bool(cfg["constraints"]["student_gradient_checkpointing"])
        student.attach_lora_adapter(student_cfg["adapter_name"], rank=student_cfg["lora_rank"],
                                    dropout=0.0, alpha=student_cfg["lora_alpha"],
                                    target=student_cfg["lora_target"])
        student.activate_single_lora(student_cfg["adapter_name"])
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(f".adapters.{student_cfg['adapter_name']}." in name)
        trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
        optimizer = FP32MasterAdamW(trainable, float(training["peak_learning_rate"]))
        optimizer.zero_grad()
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            student.load_lora_state_dict(checkpoint["adapter"], student_cfg["adapter_name"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            state = checkpoint["state"]
            report = checkpoint["report"]
            report.setdefault("resume_config_history", []).append(report.get("config"))
            report["config"] = cfg
            report["run_dir"] = str(run_dir)
            random.setstate(checkpoint["python_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        else:
            student.activate_single_lora(None)
            report["baseline_dev"] = evaluate_ifeval(
                student, tokenizer, dev_rows, verifiers,
                int(cfg["evaluation"]["batch_size"]), int(cfg["evaluation"]["max_new_tokens"]))
            student.activate_single_lora(student_cfg["adapter_name"])
            report["baseline_general"] = evaluate_general_loss(
                student, tokenizer, pools, {"chinese": 64, "english": 32})
            save_json(report_path, report)

        teacher = AutoModelForCausalLM.from_pretrained(
            str(teacher_path), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        starting_update = int(state["update"])
        target_share = composition["target_objective_share"]
        group_counts = {"student_opd": 10, "teacher_verified": 3, "general_replay": 3}
        ema_decay = float(training["objective_ema_decay"])
        while state["update"] < run_limit:
            update_began = time.perf_counter()
            update = state["update"] + 1
            batch_rows = student_schedule[state["student_cursor"]:state["student_cursor"] + 10]
            pending_samples = []
            generation_batch = int(training["student_generation_batch_size"])
            for begin in range(0, len(batch_rows), generation_batch):
                current = batch_rows[begin:begin + generation_batch]
                pending_samples.extend(generate_student_batch(
                    student, tokenizer, [row["prompt"] for row in current], training))
            student_pairs = list(zip(batch_rows, pending_samples))
            generation_seconds = time.perf_counter() - update_began
            generated_tokens = sum(len(sample["token_ids"]) for sample in pending_samples)
            language_mix = composition["general_language_cycle"][(update - 1) % len(
                composition["general_language_cycle"])]
            routes = (["student_opd"] * 10 + ["teacher_verified"] * 3
                      + ["general_chinese"] * language_mix["chinese"]
                      + ["general_english"] * language_mix["english"])
            random.shuffle(routes)
            raw_losses = collections.defaultdict(list)
            scaled_losses = collections.defaultdict(list)
            route_meta = []
            student_index = 0
            for route in routes:
                if route == "student_opd":
                    row, sample = student_pairs[student_index]; student_index += 1
                    loss, meta = on_policy_loss(
                        student, teacher, tokenizer, row, sample, training, verifiers)
                    state["student_cursor"] += 1
                    state["student_samples"] += 1
                    state["student_passed"] += int(meta["verifier_passed"])
                    state["student_loops"] += int(meta["loop"])
                    state["student_truncated"] += int(meta["hit_token_cap"])
                    group = "student_opd"
                elif route == "teacher_verified":
                    cached = teacher_schedule[state["teacher_cursor"]]
                    state["teacher_cursor"] += 1
                    loss, meta = teacher_anchor_loss(
                        student, teacher, tokenizer, cached, training, verifiers)
                    group = "teacher_verified"
                else:
                    language = route.split("_", 1)[1]
                    pool = pools[language + "_train"]
                    cursor = state["general_cursors"][language]
                    replay_row = pool[cursor % len(pool)]
                    state["general_cursors"][language] = cursor + 1
                    loss, tokens = assistant_only_ce(student, tokenizer, replay_row["messages"])
                    meta = {"id": replay_row["id"], "tokens": tokens, "language": language}
                    group = "general_replay"
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss in {route}")
                raw = float(loss.detach())
                previous_ema = state["loss_ema"].get(group, raw)
                requested_scale = float(target_share[group]) / group_counts[group] / max(previous_ema, 1e-8)
                scale = requested_scale
                scale = min(float(training["objective_scale_max"]),
                            max(float(training["objective_scale_min"]), scale))
                scaled = loss * scale
                scaled.backward()
                raw_losses[group].append(raw)
                scaled_losses[group].append(float(scaled.detach()))
                state["loss_ema"][group] = ema_decay * previous_ema + (1 - ema_decay) * raw
                route_meta.append({"route": route, "group": group, "raw_loss": raw,
                                   "scale": scale, "requested_scale": requested_scale,
                                   "scale_clipped": scale != requested_scale,
                                   "scaled_loss": float(scaled.detach()), **meta})
                del loss, scaled
            if update % int(training["gc_collect_every_updates"]) == 0:
                gc.collect()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"])))
            lr = learning_rate(update, total_updates, float(training["peak_learning_rate"]),
                               float(training["minimum_learning_rate"]),
                               int(training["warmup_optimizer_steps"]))
            optimizer.set_lr(lr); optimizer.step(); optimizer.zero_grad()
            state["update"] = update
            group_contribution = {key: sum(values) for key, values in scaled_losses.items()}
            total_contribution = sum(group_contribution.values())
            shares = {key: value / total_contribution for key, value in group_contribution.items()}
            record = {"update": update, "learning_rate": lr, "grad_norm": grad_norm,
                      "update_seconds": time.perf_counter() - update_began,
                      "generation_seconds": generation_seconds,
                      "generated_tokens": generated_tokens,
                      "generation_tokens_per_second": generated_tokens/max(generation_seconds,1e-8),
                      "raw_loss": {key: sum(value) / len(value) for key, value in raw_losses.items()},
                      "scaled_loss_sum": group_contribution, "effective_share": shares,
                      "loss_ema": dict(state["loss_ema"]), "routes": route_meta}
            report["history"].append(record)
            report["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
            report["elapsed_seconds"] = time.perf_counter() - began
            eta = report["elapsed_seconds"] / max(1, update-starting_update) * (run_limit - update)
            print(f"IFEval OPD update {update}/{run_limit} global={update}/{total_updates} "
                  f"student_seen={state['student_cursor']} pass_rate="
                  f"{state['student_passed']/state['student_samples']:.4f} "
                  f"share={shares} lr={lr:.3e} grad={grad_norm:.4f} "
                  f"peak_gpu_gb={report['peak_gpu_mb']/1024:.2f} eta_hours={eta/3600:.2f} "
                  f"seconds_per_update={record['update_seconds']:.2f} gen_tok_s={record['generation_tokens_per_second']:.1f} "
                  f"raw_loss={record['raw_loss']}", flush=True)
            if update % int(training["checkpoint_every_updates"]) == 0 or update == run_limit:
                checkpoint_file = run_dir / f"checkpoint_update_{update:05d}.pt"
                save_checkpoint(checkpoint_file, student, student_cfg["adapter_name"],
                                optimizer, state, report)
                report["latest_checkpoint"] = str(checkpoint_file)
                save_json(report_path, report)
                # Bound optimizer checkpoint storage; adapters selected during
                # evaluation are small separate files and remain preserved.
                keep = max(2, int(training.get("keep_last_checkpoints", 2)))
                checkpoints = sorted(run_dir.glob("checkpoint_update_*.pt"))
                for obsolete in checkpoints[:-keep]:
                    obsolete.unlink()
            if update % int(training.get("general_eval_every_updates", 120)) == 0 and update % updates_per_epoch != 0:
                general_check = evaluate_general_loss(student, tokenizer, pools, {"chinese":64,"english":32})
                report.setdefault("retention_checks", []).append({"update":update,"general":general_check})
                baseline = report.get("baseline_general", {})
                limit = float(cfg["evaluation"]["maximum_general_loss_increase"])
                regressed = [lang for lang in ("chinese","english")
                             if lang in baseline and general_check[lang]["loss"] > baseline[lang]["loss"]*(1+limit)]
                save_json(report_path, report)
                if regressed:
                    raise RuntimeError(f"Language retention guard stopped training at {update}: {regressed}")
            if update % updates_per_epoch == 0:
                evaluation = evaluate_ifeval(
                    student, tokenizer, dev_rows, verifiers,
                    int(cfg["evaluation"]["batch_size"]), int(cfg["evaluation"]["max_new_tokens"]))
                general = evaluate_general_loss(
                    student, tokenizer, pools, {"chinese": 64, "english": 32})
                adapter_file = run_dir / f"adapter_epoch_{update // updates_per_epoch}.pt"
                torch.save(student.get_lora_state_dict(student_cfg["adapter_name"]), adapter_file)
                report["evaluations"].append({"update": update, "ifeval_dev": evaluation,
                                              "general": general, "adapter": str(adapter_file)})
                baseline = report.get("baseline_general", {})
                limit = float(cfg["evaluation"]["maximum_general_loss_increase"])
                eligible = all(lang in baseline and general[lang]["loss"] <= baseline[lang]["loss"]*(1+limit)
                               for lang in ("chinese", "english"))
                if eligible and evaluation["accuracy"] > report.get("best_eligible_accuracy", -1):
                    report["best_eligible_accuracy"] = evaluation["accuracy"]
                    report["best_eligible_adapter"] = str(adapter_file)
                save_json(report_path, report)
                if not eligible:
                    raise RuntimeError(f"Language retention guard stopped training after epoch at {update}")

        window = report["history"][-int(training["pilot_updates"]):]
        totals = collections.defaultdict(float)
        for record in window:
            for key, value in record["scaled_loss_sum"].items():
                totals[key] += value
        denominator = sum(totals.values())
        audit_shares = {key: value / denominator for key, value in totals.items()}
        student_share = audit_shares.get("student_opd", 0.0)
        pilot_ok = (float(training["pilot_student_share_min"]) <= student_share
                    <= float(training["pilot_student_share_max"]))
        report["pilot_audit"] = {"window_updates": len(window), "effective_share": audit_shares,
                                  "student_share_in_target_band": pilot_ok,
                                  "target_band": [training["pilot_student_share_min"],
                                                  training["pilot_student_share_max"]]}
        report["status"] = "pilot_completed" if run_limit < total_updates else "completed"
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_json(report_path, report)
        print(json.dumps({"status": report["status"], "run_dir": str(run_dir),
                          "latest_checkpoint": report.get("latest_checkpoint"),
                          "pilot_audit": report["pilot_audit"],
                          "student_pass_rate": state["student_passed"] / state["student_samples"],
                          "peak_gpu_mb": report.get("peak_gpu_mb")}, ensure_ascii=False, indent=2))
    except Exception as exc:
        report["status"] = "failed"; report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        save_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
