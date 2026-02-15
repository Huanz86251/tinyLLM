"""Protocol-aligned hybrid OPD for MiniCPM3-4B -> custom 0.51B tinyLLM.

One optimizer update contains exactly 16 sequential examples:
  10 student GSM8K rollouts scored by the frozen teacher with generalized JSD;
   2 strictly answer-verified correct GSM8K paths with reasoning JSD/CE and answer-only CE;
   4 short multilingual general SFT examples with assistant-only CE.

The physical microbatch is one.  The 16 losses are divided by 16 before
backward, so the effective batch is 16 without placing both models and a large
activation batch on a 16 GB GPU.
"""
import argparse
import difflib
import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_datasets import load_local_dataset
from project_paths import path
from train import GRPO as g
from train.opd_gsm8k_stage import chunked_jsd, evaluate_indices, is_loop, learning_rate, save_json


def file_sha256(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(file):
    with Path(file).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_subsequence(values, needle, start=0):
    width = len(needle)
    for at in range(start, len(values) - width + 1):
        if values[at:at + width] == needle:
            return at
    return -1


def render_sft_with_assistant_mask(tokenizer, messages):
    """Render the platform ChatML and mark assistant content plus im_end tokens."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    ending = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    target = [False] * len(ids)
    cursor = 0
    assistant_sections = 0
    while True:
        begin = find_subsequence(ids, header, cursor)
        if begin < 0:
            break
        content_start = begin + len(header)
        content_end = find_subsequence(ids, ending, content_start)
        if content_end < 0:
            raise ValueError("assistant section has no im_end")
        for index in range(content_start, content_end + len(ending)):
            target[index] = True
        assistant_sections += 1
        cursor = content_end + len(ending)
    if assistant_sections == 0 or not any(target[1:]):
        raise ValueError("no trainable assistant tokens")
    return ids, target


def assistant_only_ce(student, tokenizer, messages):
    """Standard next-token CE, masked to assistant spans only."""
    ids, target = render_sft_with_assistant_mask(tokenizer, messages)
    tensor = torch.tensor([ids], dtype=torch.long, device="cuda")
    mask = torch.ones_like(tensor)
    output = student(input_ids=tensor, attention_mask=mask, use_cache=False,
                     force_checkpoint=False)
    logits = output["logits"][:, :-1, :]
    labels = tensor[:, 1:]
    valid = torch.tensor(target[1:], dtype=torch.bool, device="cuda").unsqueeze(0)
    loss = F.cross_entropy(logits[valid].float(), labels[valid], reduction="mean")
    tokens = int(valid.sum().item())
    del output, logits, labels, valid, tensor, mask
    return loss, tokens


def token_mask_for_char_spans(completion_ids, tokenizer, char_spans, device=None):
    """Align decoded character spans to original generated tokens without retokenizing loss."""
    ids = [int(token) for token in completion_ids]
    raw_text = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    encoded = tokenizer(
        raw_text, add_special_tokens=False, return_offsets_mapping=True)
    canonical_ids = [int(token) for token in encoded["input_ids"]]
    canonical_offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    original_offsets = [None] * len(ids)
    matcher = difflib.SequenceMatcher(
        a=canonical_ids, b=ids, autojunk=False)
    previous_a = previous_b = 0
    for block in matcher.get_matching_blocks():
        a, b, size = block.a, block.b, block.size
        if b > previous_b:
            if a > previous_a:
                gap_begin = canonical_offsets[previous_a][0]
                gap_end = canonical_offsets[a - 1][1]
            else:
                gap_begin = (canonical_offsets[a][0]
                             if a < len(canonical_offsets) else len(raw_text))
                gap_end = gap_begin
            for original_index in range(previous_b, b):
                original_offsets[original_index] = (gap_begin, gap_end)
        for delta in range(size):
            original_offsets[b + delta] = canonical_offsets[a + delta]
        previous_a, previous_b = a + size, b + size
    fallback_position = len(raw_text)
    original_offsets = [
        pair if pair is not None else (fallback_position, fallback_position)
        for pair in original_offsets]
    mask = torch.zeros((1, len(ids)), dtype=torch.bool, device=device)
    for token_index, (begin, end) in enumerate(original_offsets):
        if any(end > span_begin and begin < span_end
               for span_begin, span_end in char_spans):
            mask[0, token_index] = True
    return mask, raw_text, canonical_ids == ids


def protocol_distillation_mask(completion_ids, tokenizer, device=None):
    """Exclude every token overlapping either multi-token thought boundary."""
    ids = [int(token) for token in completion_ids]
    raw_text = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    spans = []
    metadata = {"thought_start_spans": 0, "thought_end_spans": 0,
                "protocol_tokens_masked": 0, "span_mapping": "aligned_offset_mapping"}
    for key, text in (("thought_start_spans", g.THOUGHT_START),
                      ("thought_end_spans", g.THOUGHT_END)):
        cursor = 0
        while True:
            begin = raw_text.find(text, cursor)
            if begin < 0:
                break
            spans.append((begin, begin + len(text)))
            metadata[key] += 1
            cursor = begin + len(text)
    boundary_positions, mapped_text, token_idempotent = token_mask_for_char_spans(
        ids, tokenizer, spans, device=device)
    if mapped_text != raw_text:
        raise RuntimeError("protocol span text mapping drift")
    metadata["token_idempotent"] = token_idempotent
    mask = ~boundary_positions
    metadata["protocol_tokens_masked"] = int(boundary_positions.sum().item())
    return mask, metadata


def final_answer_token_mask(completion_ids, tokenizer, device=None):
    """Locate the last boxed answer payload, including an unfinished boxed tail."""
    ids = [int(token) for token in completion_ids]
    raw_text = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    matches = list(g.BOX_RE.finditer(raw_text))
    partial = False
    if matches:
        match = matches[-1]
        span = match.span(1) if match.lastindex else match.span(0)
    else:
        marker = raw_text.rfind("\\boxed{")
        if marker < 0:
            return (torch.zeros((1, len(ids)), dtype=torch.bool, device=device),
                    {"answer_span_found": False, "partial_answer_span": False,
                     "answer_tokens": 0,
                     "answer_span_mapping": "aligned_offset_mapping"})
        span = (marker + len("\\boxed{"), len(raw_text))
        partial = True
    mask, mapped_text, token_idempotent = token_mask_for_char_spans(
        ids, tokenizer, [span], device=device)
    if mapped_text != raw_text:
        raise RuntimeError("final answer span text mapping drift")
    return mask, {"answer_span_found": True,
                  "partial_answer_span": partial,
                  "answer_tokens": int(mask.sum().item()),
                  "answer_span_mapping": "aligned_offset_mapping",
                  "token_idempotent": token_idempotent}


def sequence_logits(model, context_ids, completion_ids):
    full = torch.tensor([context_ids + completion_ids], dtype=torch.long, device="cuda")
    mask = torch.ones_like(full)
    start = len(context_ids) - 1
    end = start + len(completion_ids)
    output = model(input_ids=full, attention_mask=mask, use_cache=False,
                   **({"force_checkpoint": False} if hasattr(model, "cfg") else {}))
    return output, output["logits"][:, start:end, :], full, mask


def on_policy_math_loss(student, teacher, tokenizer, example, cfg, system_prompt,
                        precomputed_sample=None):
    """JSD on student reasoning; harden only verified-correct final answers."""
    user = g.build_gsm8k_prompt(example["question"])
    if precomputed_sample is None:
        sample = g.sample_for_grpo_manual(
            student, tokenizer, user, system_prompt, num_generations=1,
            max_new_tokens=cfg["student_rollout_max_new_tokens"],
            temperature=cfg["student_temperature"], top_p=cfg["student_top_p"],
            repetition_penalty=cfg["student_repetition_penalty"],
            no_repeat_ngram_size=cfg["student_no_repeat_ngram_size"],
        )
        completion = sample["token_ids"][0]
        text = sample["texts"][0]
        generation_batch_size = 1
    else:
        completion = list(precomputed_sample["token_ids"])
        text = str(precomputed_sample["text"])
        generation_batch_size = int(precomputed_sample["generation_batch_size"])
    if not completion:
        raise RuntimeError("student produced an empty trajectory")
    prompt = g.render_prompt_with_tokenizer(tokenizer, user, system_prompt)
    context = tokenizer.encode(prompt, add_special_tokens=False)
    with torch.no_grad():
        teacher_out, teacher_logits, teacher_full, teacher_mask = sequence_logits(
            teacher, context, completion)
        teacher_logits = teacher_logits.detach()
    student.train()
    student_out, student_logits, student_full, student_mask = sequence_logits(
        student, context, completion)
    protocol_mask, protocol_meta = protocol_distillation_mask(
        completion, tokenizer, device=student_logits.device)
    answer_mask, answer_meta = final_answer_token_mask(
        completion, tokenizer, device=student_logits.device)
    semantic_mask = protocol_mask & ~answer_mask
    semantic_jsd = chunked_jsd(
        student_logits, teacher_logits, cfg["jsd_beta"],
        cfg["distillation_temperature"], cfg["jsd_chunk_tokens"],
        token_mask=semantic_mask)
    parsed = g.parse_gsm8k_prediction(text, tokenizer)
    ground_truth = g.extract_gsm8k_gt(example["answer"])
    answer_correct = (
        parsed["final_answer"] is not None and ground_truth is not None
        and abs(parsed["final_answer"] - ground_truth) < 1e-6
        and parsed["has_box_and_valid"])
    answer_positions = answer_mask[0]
    if answer_correct and bool(answer_positions.any().item()):
        targets = torch.tensor(
            completion, dtype=torch.long, device=student_logits.device)
        answer_ce = F.cross_entropy(
            student_logits[0, answer_positions].float(),
            targets[answer_positions], reduction="mean")
    else:
        targets = None
        answer_ce = student_logits.sum() * 0.0
    answer_weight = float(cfg.get("student_correct_answer_ce_weight", 0.10))
    loss = semantic_jsd + answer_weight * answer_ce
    meta = {"tokens": len(completion),
            "generation_batch_size": generation_batch_size,
            "hit_token_cap": len(completion) >= cfg["student_rollout_max_new_tokens"],
            "loop": is_loop(text),
            "has_thought": g.THOUGHT_START in text and g.THOUGHT_END in text,
            "boxed": bool(g.BOX_RE.search(text)), "text_preview": text[:240],
            "semantic_jsd": float(semantic_jsd.detach()),
            "student_answer_correct": bool(answer_correct),
            "student_answer_ce": float(answer_ce.detach()),
            "student_answer_ce_weight": answer_weight,
            "wrong_answer_tokens_directly_trained": False,
            **protocol_meta, **answer_meta}
    del teacher_out, teacher_logits, teacher_full, teacher_mask
    del student_out, student_logits, student_full, student_mask
    del protocol_mask, answer_mask, semantic_mask, answer_positions
    del answer_ce, semantic_jsd
    if targets is not None:
        del targets
    return loss, meta


@torch.no_grad()
def sample_verified_teacher_path(teacher, tokenizer, example, teacher_cfg, system_prompt):
    """Low-temperature teacher path; accept only correct, boxed, nonempty-thought output."""
    user = g.build_gsm8k_prompt(example["question"])
    prompt = g.render_prompt_with_tokenizer(tokenizer, user, system_prompt)
    context = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([context], dtype=torch.long, device="cuda")
    attention = torch.ones_like(input_ids)
    eos = [x for x in (tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|im_end|>"))
           if x is not None and int(x) >= 0]
    ground_truth = g.extract_gsm8k_gt(example["answer"])
    attempts = []
    for attempt in range(1, teacher_cfg["maximum_attempts_per_correct_path"] + 1):
        output = teacher.generate(
            input_ids=input_ids, attention_mask=attention,
            max_new_tokens=teacher_cfg["max_new_tokens"], do_sample=True,
            temperature=teacher_cfg["temperature"], top_p=teacher_cfg["top_p"],
            repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos or tokenizer.eos_token_id,
        )
        completion = output[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(completion, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip()
        parsed = g.parse_gsm8k_prediction(text, tokenizer)
        correct = (parsed["final_answer"] is not None and ground_truth is not None and
                   abs(parsed["final_answer"] - ground_truth) < 1e-6)
        valid_thought = parsed["has_cot_tags"] and parsed["cot_token_len"] > 0
        accepted = (correct and (parsed["has_box_and_valid"] or not teacher_cfg["require_boxed_answer"])
                    and (valid_thought or not teacher_cfg["require_nonempty_thought"])
                    and not is_loop(text))
        attempts.append({"attempt": attempt, "correct": bool(correct),
                         "boxed": parsed["has_box_and_valid"], "thought_tokens": parsed["cot_token_len"],
                         "tokens": len(completion), "loop": is_loop(text)})
        if accepted:
            return context, completion, text, attempts
    raise RuntimeError("teacher failed verified path filter: " + json.dumps(attempts))


def teacher_math_loss(student, teacher, tokenizer, example, training_cfg, teacher_cfg, system_prompt):
    """Grounded teacher path: generalized JSD plus hard-label next-token CE."""
    context, completion, text, attempts = sample_verified_teacher_path(
        teacher, tokenizer, example, teacher_cfg, system_prompt)
    with torch.no_grad():
        teacher_out, teacher_logits, teacher_full, teacher_mask = sequence_logits(
            teacher, context, completion)
        teacher_logits = teacher_logits.detach()
    student.train()
    student_out, student_logits, student_full, student_mask = sequence_logits(
        student, context, completion)
    targets = torch.tensor(completion, dtype=torch.long, device="cuda")
    jsd = chunked_jsd(student_logits, teacher_logits, training_cfg["jsd_beta"],
                      training_cfg["distillation_temperature"], training_cfg["jsd_chunk_tokens"])
    ce = F.cross_entropy(student_logits[0].float(), targets, reduction="mean")
    loss = training_cfg["teacher_math_jsd_weight"] * jsd + training_cfg["teacher_math_ce_weight"] * ce
    meta = {"tokens": len(completion), "loop": False, "has_thought": True, "boxed": True,
            "trajectory_source": "verified_teacher_generation", "attempts": attempts,
            "jsd": float(jsd.detach()), "ce": float(ce.detach()),
            "text_preview": text[:240]}
    del teacher_out, teacher_logits, teacher_full, teacher_mask
    del student_out, student_logits, student_full, student_mask, targets, jsd, ce
    return loss, meta


class FP32MasterAdamW:
    """BF16 LoRA in the model, FP32 master weights and Adam states."""
    def __init__(self, parameters, lr):
        self.sources = list(parameters)
        self.master = [torch.nn.Parameter(p.detach().float().clone(), requires_grad=True)
                       for p in self.sources]
        self.optimizer = torch.optim.AdamW(self.master, lr=lr, weight_decay=0.0)

    def zero_grad(self):
        for source in self.sources:
            source.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        for source, master in zip(self.sources, self.master):
            master.grad = None if source.grad is None else source.grad.detach().float().clone()
        self.optimizer.step()
        with torch.no_grad():
            for source, master in zip(self.sources, self.master):
                source.copy_(master.to(source.dtype))

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(),
                "master": [x.detach().cpu() for x in self.master]}

    def load_state_dict(self, state):
        with torch.no_grad():
            for target, saved in zip(self.master, state["master"]):
                target.copy_(saved.to(target.device))
            for source, master in zip(self.sources, self.master):
                source.copy_(master.to(source.dtype))
        self.optimizer.load_state_dict(state["optimizer"])


def validate_config(cfg):
    composition = cfg["composition"]
    math_total = (composition["student_on_policy_gsm8k"] +
                  composition["teacher_correct_gsm8k"])
    general_total = composition["general_sft_per_update"]
    total = math_total + general_total
    if total != composition["effective_global_batch"] or total != 16:
        raise ValueError("composition must be 10 student + 2 teacher + 4 general = 16")
    cycle = composition["general_language_cycle"]
    if not cycle:
        raise ValueError("general_language_cycle cannot be empty")
    language_totals = {"chinese": 0, "english": 0}
    for item in cycle:
        if set(item) != set(language_totals):
            raise ValueError("each language cycle item must contain chinese and english")
        if sum(item.values()) != general_total:
            raise ValueError("each language cycle item must contain four general rows")
        for language in language_totals:
            language_totals[language] += item[language]
    if language_totals["chinese"] != 2 * language_totals["english"]:
        raise ValueError("aggregate general replay ratio must be exactly 2:1 Chinese/English")
    training = cfg["training"]
    if "round_optimizer_updates" in training:
        round_updates = training["round_optimizer_updates"]
        if len(round_updates) != training["rounds"] or any(x <= 0 for x in round_updates):
            raise ValueError("round_optimizer_updates must contain one positive value per round")
    if cfg["training"]["physical_microbatch"] != 1:
        raise ValueError("16 GB profile requires physical_microbatch=1")
    teacher_weight_sum = (
        cfg["training"]["teacher_math_jsd_weight"]
        + cfg["training"]["teacher_math_ce_weight"]
        + cfg["training"].get("teacher_math_answer_ce_weight", 0.0))
    if not math.isclose(teacher_weight_sum, 1.0):
        raise ValueError("teacher math JSD, trace CE and answer CE weights must sum to one")


def verify_protocol(cfg):
    prompt_cfg = json.loads(path("configs/prompts.json").read_text(encoding="utf-8-sig"))
    current = prompt_cfg["gsm8k"]
    protocol = cfg["platform_protocol"]
    if current["system"] != protocol["system_prompt"] or current["suffix"] != protocol["math_suffix"]:
        raise RuntimeError("training protocol drifted from configs/prompts.json:gsm8k")


def load_general_pools(cfg):
    directory = path(cfg["dataset"]["general_sft"]["output_dir"])
    manifest_file = directory / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError("run tools/prepare_opd_hybrid_multilingual.py --filter-review first")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    pools = {}
    for source in ("chinese", "english"):
        for split in ("train", "eval"):
            key = f"{source}_{split}"
            rec = manifest["files"][key]
            file = Path(rec["path"])
            if not file.is_file():
                portable_name = Path(str(rec["path"]).replace("\\", "/")).name
                file = directory / portable_name
            if file_sha256(file) != rec["sha256"]:
                raise RuntimeError(f"general SFT checksum mismatch: {file}")
            pools[key] = read_jsonl(file)
    return pools, manifest


@torch.no_grad()
def evaluate_general_loss(student, tokenizer, pools, rows_per_language):
    was_training = student.training
    student.eval()
    result = {}
    for source in ("chinese", "english"):
        losses, tokens = [], []
        limit = rows_per_language[source]
        for row in pools[source + "_eval"][:limit]:
            loss, count = assistant_only_ce(student, tokenizer, row["messages"])
            losses.append(float(loss.detach()))
            tokens.append(count)
            del loss
        result[source] = {"loss": sum(losses) / len(losses), "rows": len(losses),
                          "assistant_tokens": sum(tokens)}
    total_rows = sum(result[x]["rows"] for x in ("chinese", "english"))
    result["mean_loss"] = sum(result[x]["loss"] * result[x]["rows"]
                              for x in ("chinese", "english")) / total_rows
    if was_training:
        student.train()
    return result


def save_checkpoint(file, student, adapter_name, optimizer, state, report):
    payload = {"schema_version": 1, "adapter_name": adapter_name,
               "adapter": student.get_lora_state_dict(adapter_name),
               "optimizer": optimizer.state_dict(), "state": state, "report": report,
               "python_random_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
               "cuda_rng_state": torch.cuda.get_rng_state_all()}
    temp = Path(str(file) + ".tmp")
    torch.save(payload, temp)
    temp.replace(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(path("configs/opd_hybrid_v3.json")))
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    validate_config(cfg)
    verify_protocol(cfg)
    pools, manifest = load_general_pools(cfg)
    training = cfg["training"]
    composition = cfg["composition"]
    system_prompt = cfg["platform_protocol"]["system_prompt"]
    if "round_optimizer_updates" in training:
        round_updates = list(training["round_optimizer_updates"])
    else:
        round_updates = [training["optimizer_updates_per_round"]] * training["rounds"]
    round_boundaries = []
    for count in round_updates:
        round_boundaries.append(count + (round_boundaries[-1] if round_boundaries else 0))
    total_updates = round_boundaries[-1]
    math_per_update = (composition["student_on_policy_gsm8k"] +
                       composition["teacher_correct_gsm8k"])
    if args.dry_run:
        gsm_rows = len(load_local_dataset("gsm8k", "train"))
        available_math = gsm_rows - cfg["dataset"]["gsm8k"]["dev_size"]
        planned_math = total_updates * math_per_update
        print(json.dumps({"status": "ready", "composition": composition,
                          "round_optimizer_updates": round_updates,
                          "total_optimizer_updates": total_updates,
                          "gsm8k_available_after_dev": available_math,
                          "planned_math_slots": planned_math,
                          "full_coverage": planned_math >= available_math,
                          "tail_repeats": max(0, planned_math - available_math),
                          "protocol": cfg["platform_protocol"],
                          "general_files": manifest["files"]}, ensure_ascii=False, indent=2))
        return
    random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    torch.cuda.manual_seed_all(training["seed"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = cfg.get("run_name", "opd_hybrid_v3")
    run_dir = path("runs/training") / run_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report = {"status": "starting", "config": cfg, "general_manifest": manifest,
              "run_dir": str(run_dir), "history": [], "round_evaluations": [],
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "auto_activated": False}
    save_json(report_path, report)
    try:
        gsm_train = load_local_dataset("gsm8k", "train")
        gsm_test = load_local_dataset("gsm8k", "test")
        order = list(range(len(gsm_train)))
        random.Random(cfg["dataset"]["gsm8k"]["split_seed"]).shuffle(order)
        dev_size = cfg["dataset"]["gsm8k"]["dev_size"]
        dev_indices = order[:dev_size]
        math_order = order[dev_size:]
        needed_math = total_updates * math_per_update
        full_coverage = bool(training.get("full_gsm8k_train_coverage", False))
        if full_coverage and needed_math < len(math_order):
            raise RuntimeError("full GSM8K coverage requested but schedule is too short")
        if needed_math > len(math_order):
            repeat_count = needed_math - len(math_order)
            if not full_coverage or repeat_count > len(math_order):
                raise RuntimeError("math schedule exceeds unique GSM8K prompts")
            math_schedule = math_order + math_order[:repeat_count]
        else:
            repeat_count = 0
            math_schedule = math_order[:needed_math]
        report["coverage_plan"] = {"available_after_dev": len(math_order),
                                   "planned_math_slots": needed_math,
                                   "unique_prompts_covered": len(set(math_schedule)),
                                   "tail_repeats": repeat_count,
                                   "round_optimizer_updates": round_updates}
        save_json(report_path, report)
        print(json.dumps({"status": "training_plan", **report["coverage_plan"],
                          "run_dir": str(run_dir)}, ensure_ascii=False, indent=2), flush=True)
        teacher_path = path(cfg["teacher"]["path"])
        tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), trust_remote_code=True,
                                                  local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        student_cfg = cfg["student"]
        student = g.load_tinyllm_from_ckpt(str(path(student_cfg["path"])), tokenizer, strict=True)
        student.attach_lora_adapter(student_cfg["adapter_name"], rank=student_cfg["lora_rank"],
                                    dropout=0.0, alpha=student_cfg["lora_alpha"],
                                    target=student_cfg["lora_target"])
        student.activate_single_lora(student_cfg["adapter_name"])
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(f".adapters.{student_cfg['adapter_name']}." in name)
        trainable = [p for p in student.parameters() if p.requires_grad]
        optimizer = FP32MasterAdamW(trainable, training["peak_learning_rate"])
        optimizer.zero_grad()
        state = {"update": 0, "math_cursor": 0, "general_cursors": {"chinese": 0, "english": 0}}
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            student.load_lora_state_dict(checkpoint["adapter"], student_cfg["adapter_name"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            state = checkpoint["state"]
            report = checkpoint["report"]
            report["run_dir"] = str(run_dir)
            random.setstate(checkpoint["python_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            state.setdefault("general_cursors", {})
            for language in ("chinese", "english"):
                state["general_cursors"].setdefault(language, 0)
        else:
            student.activate_single_lora(None)
            baseline_math = evaluate_indices(student, tokenizer, gsm_train, dev_indices,
                                             cfg["evaluation"]["max_new_tokens"],
                                             cfg["evaluation"]["batch_size"])
            student.activate_single_lora(student_cfg["adapter_name"])
            baseline_general = evaluate_general_loss(student, tokenizer, pools,
                                                     cfg["evaluation"]["general_eval_rows_per_language"])
            report["baseline_math_dev"] = baseline_math
            report["baseline_general"] = baseline_general
            save_json(report_path, report)
        teacher = AutoModelForCausalLM.from_pretrained(
            str(teacher_path), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        best = report.get("best_candidate")
        while state["update"] < total_updates:
            update = state["update"] + 1
            language_mix = composition["general_language_cycle"][
                (update - 1) % len(composition["general_language_cycle"])]
            route_plan = (["student_math"] * composition["student_on_policy_gsm8k"] +
                          ["teacher_math"] * composition["teacher_correct_gsm8k"] +
                          ["general_chinese"] * language_mix["chinese"] +
                          ["general_english"] * language_mix["english"])
            random.shuffle(route_plan)
            route_losses = {name: [] for name in set(route_plan)}
            route_meta = []
            for route in route_plan:
                if route in ("student_math", "teacher_math"):
                    example_index = math_schedule[state["math_cursor"]]
                    state["math_cursor"] += 1
                    example = gsm_train[example_index]
                    if route == "student_math":
                        loss, meta = on_policy_math_loss(student, teacher, tokenizer, example,
                                                         training, system_prompt)
                    else:
                        loss, meta = teacher_math_loss(student, teacher, tokenizer, example,
                                                       training, cfg["teacher"], system_prompt)
                    meta["gsm8k_index"] = example_index
                else:
                    source = route.split("_", 1)[1]
                    pool = pools[source + "_train"]
                    cursor = state["general_cursors"][source]
                    row = pool[cursor % len(pool)]
                    state["general_cursors"][source] = cursor + 1
                    loss, tokens = assistant_only_ce(student, tokenizer, row["messages"])
                    meta = {"id": row["id"], "tokens": tokens, "loop": False,
                            "has_thought": True, "boxed": False}
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss in route {route}")
                (loss / composition["effective_global_batch"]).backward()
                route_losses[route].append(float(loss.detach()))
                route_meta.append({"route": route, **meta})
                del loss
                gc.collect()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, training["max_grad_norm"]))
            lr = learning_rate(update, total_updates, training["peak_learning_rate"],
                               training["minimum_learning_rate"], training["warmup_optimizer_steps"])
            optimizer.set_lr(lr)
            optimizer.step()
            optimizer.zero_grad()
            state["update"] = update
            round_number = next(i + 1 for i, boundary in enumerate(round_boundaries)
                                if update <= boundary)
            record = {"update": update, "round": round_number,
                      "learning_rate": lr, "grad_norm": grad_norm,
                      "loss": {name: sum(values) / len(values) for name, values in route_losses.items()},
                      "routes": route_meta}
            report["history"].append(record)
            report["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            report["elapsed_seconds"] = time.perf_counter() - began
            elapsed = report["elapsed_seconds"]
            eta_seconds = elapsed / update * (total_updates - update)
            round_start = 0 if round_number == 1 else round_boundaries[round_number - 2]
            round_position = update - round_start
            print(f"round {round_number}/{len(round_updates)} update {round_position}/{round_updates[round_number - 1]} "
                  f"total {update}/{total_updates} eta_hours={eta_seconds / 3600:.2f} "
                  f"lr={lr:.3e} grad={grad_norm:.4f} loss={record['loss']}", flush=True)
            if update % training["checkpoint_every_updates"] == 0:
                checkpoint_file = run_dir / f"checkpoint_update_{update:04d}.pt"
                save_checkpoint(checkpoint_file, student, student_cfg["adapter_name"],
                                optimizer, state, report)
                report["latest_checkpoint"] = str(checkpoint_file)
                save_json(report_path, report)
            if update in round_boundaries:
                round_number = round_boundaries.index(update) + 1
                math_eval = evaluate_indices(student, tokenizer, gsm_train, dev_indices,
                                             cfg["evaluation"]["max_new_tokens"],
                                             cfg["evaluation"]["batch_size"])
                general_eval = evaluate_general_loss(student, tokenizer, pools,
                                                     cfg["evaluation"]["general_eval_rows_per_language"])
                adapter_file = run_dir / f"adapter_round_{round_number}.pt"
                torch.save(student.get_lora_state_dict(student_cfg["adapter_name"]), adapter_file)
                base_math = report["baseline_math_dev"]
                base_general = report["baseline_general"]
                general_limit = base_general["mean_loss"] * (1 + cfg["evaluation"]["maximum_general_loss_increase"])
                eligible = (math_eval["correct"] > base_math["correct"] and
                            math_eval["loop_rate"] <= base_math["loop_rate"] and
                            general_eval["mean_loss"] <= general_limit)
                item = {"round": round_number, "after_updates": update, "adapter": str(adapter_file),
                        "math_dev": math_eval, "general": general_eval, "eligible": eligible}
                report["round_evaluations"].append(item)
                if eligible and (best is None or math_eval["correct"] > best["math_dev"]["correct"] or
                                 (math_eval["correct"] == best["math_dev"]["correct"] and
                                  general_eval["mean_loss"] < best["general"]["mean_loss"])):
                    best = item
                    report["best_candidate"] = best
                save_json(report_path, report)
                print(json.dumps({"status": "round_completed", "round": round_number,
                                  "math_dev": math_eval, "general": general_eval,
                                  "eligible": eligible, "adapter": str(adapter_file)},
                                 ensure_ascii=False, indent=2), flush=True)
        if best:
            adapter_state = torch.load(best["adapter"], map_location="cpu")
            student.load_lora_state_dict(adapter_state, student_cfg["adapter_name"], strict=False)
            confirmation_start = cfg["dataset"]["gsm8k"]["confirmation_start"]
            confirmation_indices = list(range(confirmation_start, confirmation_start +
                                               cfg["dataset"]["gsm8k"]["confirmation_size"]))
            student.activate_single_lora(None)
            confirmation_base = evaluate_indices(student, tokenizer, gsm_test, confirmation_indices,
                                                 cfg["evaluation"]["max_new_tokens"],
                                                 cfg["evaluation"]["batch_size"])
            student.activate_single_lora(student_cfg["adapter_name"])
            confirmation_model = evaluate_indices(student, tokenizer, gsm_test, confirmation_indices,
                                                  cfg["evaluation"]["max_new_tokens"],
                                                  cfg["evaluation"]["batch_size"])
            report["official_test_confirmation"] = {"indices": confirmation_indices,
                                                     "baseline": confirmation_base,
                                                     "candidate": confirmation_model}
        report["status"] = "completed"
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["eligible_for_demo"] = bool(best)
        save_json(report_path, report)
        save_json(path("logs/history") / f"{run_name}_latest.json", report)
        print(json.dumps({"status": report["status"], "best": best,
                          "confirmation": report.get("official_test_confirmation"),
                          "peak_gpu_mb": report.get("peak_gpu_mb")}, ensure_ascii=False, indent=2))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        save_json(report_path, report)
        raise


if __name__ == "__main__":
    main()

