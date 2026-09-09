"""One resident model family, real LoRA activation, and cancellable token streaming."""
from pathlib import Path
import gc, hashlib, json, queue, re, threading, time, uuid
import torch
from PIL import Image, ImageOps
from inference.runtime import load_model, checked_adapter, registry, path
from inference.decoding import (SafeDecoder, trim_history, completed_boxed_answer,
                                sanitize_generated_text)

MODES = {
    "chat": {"label": "普通聊天", "model": "sft_base", "family": "text"},
    "math": {"label": "数学推理", "model": "sft_base", "family": "text", "max_new_tokens": 640},
    "opd": {"label": "OPD 对齐", "model": "opd_gsm8k_best", "family": "text", "max_new_tokens": 640},
    "arc": {"label": "科学 · ARC", "model": "arc_best", "family": "text"},
    "vision": {"label": "图片问答", "model": "vlm", "family": "vision"},
}
class BusyError(RuntimeError):
    pass

class ChatEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self.model = self.tokenizer = self.vip = None
        self.family = self.mode = self.model_id = None
        self.info = {}
        self.arc_loaded = self.opd_loaded = False
        self.opd_adapter_name = None
        self.vision_cache = None
        self._stop = threading.Event()
        self._state = {"busy": False, "phase": "等待加载 SFT", "request_id": None,
                       "error": None, "loaded": False}
        self.base_loads = 0

    def _state_update(self, **values):
        with self._meta_lock:
            self._state.update(values)

    def status(self):
        with self._meta_lock:
            result = dict(self._state)
        available = {}
        records = registry()
        for name, spec in MODES.items():
            model_id = self._mode_model_id(name, records)
            rec = records.get(model_id)
            if rec is None:
                available[name] = False
                continue
            base_ok = (path(rec["base"]) / "model.safetensors").is_file()
            adapter_ok = "adapter" not in rec or path(rec["adapter"]["path"]).is_file()
            available[name] = base_ok and adapter_ok
        result.update(mode=self.mode, family=self.family, model_id=self.model_id,
                      mode_available=available, base_loads=self.base_loads,
                      active_adapter=("arc" if self.mode == "arc" else
                                      self.opd_adapter_name if self.mode == "opd" else
                                      "vlm_sft" if self.family == "vision" else None),
                      arc_cached=self.arc_loaded, opd_cached=self.opd_loaded,
                      device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        if torch.cuda.is_available():
            result.update(gpu_allocated_mb=round(torch.cuda.memory_allocated()/2**20, 1),
                          gpu_reserved_mb=round(torch.cuda.memory_reserved()/2**20, 1))
        return result

    def _phase(self, text, emit=None):
        self._state_update(phase=text)
        if emit:
            emit({"type": "status", "message": text})

    @staticmethod
    def _mode_model_id(mode, records=None):
        records = records or registry()
        if mode == "opd" and "opd_ifeval_best" in records:
            return "opd_ifeval_best"
        return MODES[mode]["model"]

    def _release(self, emit=None):
        self._phase("正在释放上一套模型的显存", emit)
        self.model = self.tokenizer = self.vip = self.vision_cache = None
        self.family = self.mode = self.model_id = None
        self.info = {}
        self.arc_loaded = self.opd_loaded = False
        self.opd_adapter_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._state_update(loaded=False)

    def _select(self, mode, emit=None):
        if mode not in MODES:
            raise ValueError("未知模式")
        family = MODES[mode]["family"]
        requested_model_id = self._mode_model_id(mode)
        load_model_id = "sft_base" if family == "text" else requested_model_id
        use_tf32 = family == "text"
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        if self.family != family or self.model is None or self.model_id != load_model_id:
            if self.model is not None:
                self._release(emit)
            self._phase("正在加载 SFT 文字底座" if family == "text" else "正在加载视觉底座与视觉 LoRA", emit)
            self.model, self.tokenizer, self.info = load_model(
                load_model_id,
                "fp32" if family == "text" else "bf16")
            self.family = family
            self.model_id = load_model_id
            self.base_loads += 1
        if family == "text":
            if mode == "arc":
                if not self.arc_loaded:
                    self._phase("正在热加载 ARC 峰值 LoRA", emit)
                    adapter = registry()["arc_best"]["adapter"]
                    self.model.attach_lora_adapter(adapter_name=adapter["name"], rank=adapter["rank"],
                                                   dropout=adapter["dropout"], alpha=adapter["alpha"],
                                                   target=adapter["target"])
                    count = checked_adapter(self.model, path(adapter["path"]), adapter["name"])
                    if count != 280:
                        raise RuntimeError(f"ARC adapter tensor count changed: {count}")
                    self.model.eval()
                    self.arc_loaded = True
                self.model.activate_single_lora("arc")
            elif mode == "opd":
                if not self.opd_loaded:
                    self._phase("正在热加载 OPD LoRA", emit)
                    adapter = registry()[requested_model_id]["adapter"]
                    self.model.attach_lora_adapter(adapter_name=adapter["name"], rank=adapter["rank"],
                                                   dropout=adapter["dropout"], alpha=adapter["alpha"],
                                                   target=adapter["target"])
                    checked_adapter(self.model, path(adapter["path"]), adapter["name"])
                    self.model.eval()
                    self.opd_loaded = True
                    self.opd_adapter_name = adapter["name"]
                self.model.activate_single_lora(self.opd_adapter_name)
            else:
                self.model.activate_single_lora(None)
        elif self.vip is None:
            self._phase("正在加载本地视觉塔", emit)
            from eval import vllm as legacy
            tower = path(registry()[requested_model_id]["vision_tower"])
            legacy.HF_CACHE_DIR = str(path("cache/huggingface"))
            device = next(self.model.parameters()).device
            self.vip = legacy.VIPRuntime(str(tower), device,
                                          torch.float16 if device.type == "cuda" else torch.float32, True)
            self.vip.load()
        self.model.eval()
        self.mode = mode
        self._state_update(loaded=True, error=None)
        self._phase(MODES[mode]["label"] + "已就绪", emit)
        return self.status()

    def select(self, mode):
        if not self._lock.acquire(blocking=False):
            raise BusyError("模型正在加载或生成，请稍候")
        self._state_update(busy=True, request_id=None, error=None)
        try:
            with torch.inference_mode():
                return self._select(mode)
        except Exception as exc:
            self._release()
            self._state_update(error=str(exc), phase="加载失败，可以重试")
            raise
        finally:
            self._state_update(busy=False)
            self._lock.release()

    def unload(self):
        if not self._lock.acquire(blocking=False):
            raise BusyError("请先停止当前回答")
        try:
            self._release()
            self._state_update(phase="模型已卸载", error=None)
        finally:
            self._lock.release()

    def cancel(self, request_id):
        with self._meta_lock:
            matches = self._state["request_id"] == request_id
        if matches:
            self._stop.set()
        return matches

    def _vision(self, filename, emit):
        from eval import vllm as legacy
        filename = Path(filename)
        key = hashlib.sha256(filename.read_bytes()).hexdigest()
        if self.vision_cache and self.vision_cache[0] == key:
            self._phase("复用这张图片的视觉特征", emit)
            return self.vision_cache[1], True
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        with Image.open(filename) as src:
            im = ImageOps.exif_transpose(src).convert("RGB")
        names, images, meta = legacy.build_views_and_meta(im)
        example = {"vision_meta": meta, "image_files": names}
        feats, masks, poses, offsets = [], [], [], []
        for i, name in enumerate(names):
            if self._stop.is_set():
                raise InterruptedError("已停止")
            self._phase(f"正在读取图片 · 视图 {i+1}/{len(names)}", emit)
            feat = self.vip.encode_one(images[name])
            mask, _ = legacy.compute_masks_for_image(example, name)
            pos, off = legacy.compute_global_pos_for_image(example, name, mask)
            feats.append(feat)
            masks.append(torch.from_numpy(mask).bool())
            poses.append(torch.from_numpy(pos.astype("int64")))
            offsets.append(torch.from_numpy(off))
        data = {"vision_feats": torch.stack(feats)[None].to(device=device, dtype=dtype),
                "vision_mask": torch.stack(masks)[None].to(device),
                "global_pos": torch.stack(poses)[None].to(device),
                "global_off": torch.stack(offsets)[None].to(device=device, dtype=torch.float16)}
        self.vision_cache = key, data
        return data, False

    def _prompt(self, messages, mode, has_image, max_tokens):
        def safe_text(value):
            return re.sub(r"<\|im_(?:start|end)\|>|<img>", "", value)
        rows = []
        self._context_cleanups = []
        for index, message in enumerate(messages):
            content = safe_text(message["content"])
            if message["role"] == "assistant":
                clean_mode = "vision" if MODES[mode]["family"] == "vision" else mode
                content = sanitize_generated_text(content, clean_mode)
                content, cleanup = trim_history(content)
                if cleanup:
                    self._context_cleanups.append({"message_index": index, **cleanup})
            rows.append({"role": message["role"], "content": content})
        system = None
        if MODES[mode]["family"] == "vision":
            latest_user = rows[-1]["content"] if rows else ""
            language = "Chinese" if re.search(r"[\u4e00-\u9fff]", latest_user) else "English"
            system = ("Answer only from clearly visible image evidence. If a detail is unreadable or uncertain, "
                      "say so instead of guessing. For a chart, table, diagram, or document, identify its type first "
                      "and use only legible labels and values. Keep the answer concise, normally one to four sentences. "
                      "Use plain text or Markdown, never HTML/XML or image placeholder tokens. "
                      f"Respond in {language}.")
        if mode in ("math", "opd", "arc"):
            specs = json.loads(path("configs/prompts.json").read_text(encoding="utf-8"))
            rec = specs["gsm8k" if mode in ("math", "opd") else "arc"]
            system = rec["system"]
            rows[-1]["content"] += rec["suffix"]
        budget = min(4096, int(self.model.cfg.max_position_embeddings)) - max_tokens - (128 if has_image else 0)
        dropped = 0
        while rows:
            parts = [f"<|im_start|>system\n{system}<|im_end|>\n"] if system else []
            first_image = has_image
            for row in rows:
                content = row["content"]
                if first_image and row["role"] == "user":
                    content = "<img>\n" + content
                    first_image = False
                parts.append(f'<|im_start|>{row["role"]}\n{content}<|im_end|>\n')
            prompt = "".join(parts) + "<|im_start|>assistant\n"
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) <= budget:
                return prompt, ids, dropped
            if len(rows) == 1:
                raise ValueError("本次问题太长，请缩短后再发送")
            # Retain complete recent turns, always starting on a user message.
            rows.pop(0)
            dropped += 1
            while len(rows) > 1 and rows[0]["role"] != "user":
                rows.pop(0)
                dropped += 1
        raise ValueError("没有可用的聊天内容")

    def start(self, request_id, mode, messages, image_file=None, max_tokens=2048):
        if not self._lock.acquire(blocking=False):
            raise BusyError("模型正在加载或生成，请稍候")
        events = queue.Queue()
        self._stop = threading.Event()
        self._state_update(busy=True, request_id=request_id, error=None)
        thread = threading.Thread(target=self._work,
                                  args=(events, request_id, mode, messages, image_file, max_tokens),
                                  daemon=True)
        thread.start()
        return events

    def _work(self, events, request_id, mode, messages, image_file, max_tokens):
        emit = events.put
        started = time.perf_counter()
        final = None
        output = None
        try:
            with torch.inference_mode():
                emit({"type": "accepted", "request_id": request_id})
                if image_file is not None and MODES[mode]["family"] != "vision":
                    mode = "vision"
                if MODES[mode]["family"] == "vision" and image_file is None:
                    raise ValueError("请先添加图片")
                requested_max_tokens = max_tokens
                max_tokens = min(max_tokens, MODES[mode].get("max_new_tokens", max_tokens))
                self._select(mode, emit)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                if self._stop.is_set():
                    raise InterruptedError("已停止")
                vision = None
                cached = False
                if image_file:
                    vision, cached = self._vision(image_file, emit)
                prompt, ids, dropped = self._prompt(messages, mode, bool(image_file), max_tokens)
                emit({"type": "context", "input_tokens": len(ids), "dropped_messages": dropped,
                      "image_cached": cached, "mode": mode, "model": MODES[mode]["model"],
                      "max_tokens": max_tokens,
                      "cleaned_history_messages": len(self._context_cleanups)})
                self._phase("正在处理上下文，等待第一个 token", emit)
                output = self._decode(ids, vision, mode, max_tokens, started, emit)
                record = {**output, "interface": "chat_stream", "request_id": request_id,
                          "mode": mode, "model_id": MODES[mode]["model"], "messages": messages,
                          "prepared_prompt": prompt, "input_tokens": len(ids), "max_tokens": max_tokens,
                          "requested_max_tokens": requested_max_tokens,
                          "dropped_messages": dropped, "context_cleanups": self._context_cleanups,
                          "image": str(image_file) if image_file else None,
                          "image_cached": cached, "base": self.info["base"],
                          "adapter": registry()[MODES[mode]["model"]].get("adapter"),
                          "strict_base_load": True, "base_loads": self.base_loads,
                          "base_instance": id(self.model),
                          "precision": self.info["precision"],
                          "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                          "total_seconds": time.perf_counter()-started}
                if torch.cuda.is_available():
                    record["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated()/2**20, 1)
                folder = path("logs/local_runs")
                filename = folder / ("chat_" + time.strftime("%Y%m%d_%H%M%S") + "_" + request_id[:8] + ".json")
                filename.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                final = {"type": "done", **output, "total_seconds": record["total_seconds"],
                         "mode": mode, "model": MODES[mode]["model"], "log": str(filename),
                         "base_loads": self.base_loads, "base_instance": id(self.model),
                         "image_cached": cached, "dropped_messages": dropped}
                # Drop per-request feature references before a later family switch.
                del vision
        except InterruptedError:
            final = {"type": "done", "text": "", "tokens": 0, "cancelled": True,
                     "mode": mode, "total_seconds": time.perf_counter()-started}
        except Exception as exc:
            self._state_update(error=str(exc))
            final = {"type": "error", "message": str(exc)}
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                self._release()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._state_update(busy=False, request_id=None,
                               phase="已停止" if self._stop.is_set() else ("就绪" if final and final["type"] == "done" else "本次未完成，可重试"))
            self._lock.release()
            emit(final or {"type": "error", "message": "本次请求未完成"})

    def _decode(self, ids, vision, mode, max_tokens, requested_at, emit):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        decode_mode = "vision" if MODES[mode]["family"] == "vision" else ("math" if mode == "opd" else mode)
        decoder = SafeDecoder(self.tokenizer, device, decode_mode)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = []
        text = ""
        first = None
        began = time.perf_counter()
        stop_reason = "length"
        protection = None
        emit({"type": "decoding", **decoder.metadata()})
        with torch.autocast(device_type=device.type, dtype=dtype if dtype != torch.float32 else torch.bfloat16,
                            enabled=device.type == "cuda" and dtype != torch.float32):
            if self._stop.is_set():
                raise InterruptedError("已停止")
            out = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             use_cache=True, past_states=None, **(vision or {}))
            past = out["past_states"]
            logits = out["logits"][:, -1, :]
            del out
            for step in range(max_tokens):
                if self._stop.is_set():
                    stop_reason = "cancelled"
                    break
                if time.perf_counter()-began > decoder.config["max_generation_seconds"]:
                    stop_reason = "time_limit"
                    protection = {"kind": "generation_time_limit"}
                    break
                logits = logits.clone()
                eos = self.tokenizer.eos_token_id
                pad = self.tokenizer.pad_token_id
                if pad is not None and pad != eos:
                    logits[:, pad] = -float("inf")
                for name in ("unk_token_id", "bos_token_id"):
                    value = getattr(self.model.cfg, name, None)
                    if value is not None and value not in decoder.stop_ids and 0 <= int(value) < logits.shape[-1]:
                        logits[:, int(value)] = -float("inf")
                im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                if im_start is not None and im_start not in decoder.stop_ids and 0 <= im_start < logits.shape[-1]:
                    logits[:, im_start] = -float("inf")
                value, problem = decoder.choose(logits, generated)
                if problem:
                    stop_reason = "invalid_logits"
                    protection = problem
                    break
                token = torch.tensor([[value]], dtype=torch.long, device=device)
                generated.append(value)
                now = time.perf_counter()
                if first is None:
                    first = now
                    self._state_update(phase="正在生成回答")
                raw_candidate = self.tokenizer.decode(generated, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)
                candidate = sanitize_generated_text(raw_candidate, mode)
                protection = decoder.inspect(raw_candidate, generated)
                if protection:
                    stop_reason = "repetition"
                    text = sanitize_generated_text(
                        raw_candidate[:protection["keep_chars"]], mode
                    ).rstrip()
                    emit({"type": "guard", "reason": "repetition",
                          "message": "检测到循环重复，已停止生成并收起重复尾部。请核对已有内容，或换一种问法。"})
                    break
                if candidate != text and not candidate.endswith("\ufffd"):
                    text = candidate
                    emit({"type": "token", "text": text, "tokens": len(generated),
                          "first_token_seconds": first-began,
                          "first_token_total_seconds": first-requested_at,
                          "generation_seconds": now-began})
                if value in decoder.stop_ids:
                    stop_reason = "eos"
                    break
                if mode in ("math", "opd", "arc") and completed_boxed_answer(raw_candidate):
                    stop_reason = "boxed"
                    break
                if step+1 == max_tokens:
                    break
                out = self.model(input_ids=token, attention_mask=torch.ones_like(token),
                                 use_cache=True, past_states=past)
                past = out["past_states"]
                logits = out["logits"][:, -1, :]
                del out
        elapsed = time.perf_counter()-began
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False).strip()
        if stop_reason != "repetition":
            text = sanitize_generated_text(raw_text, mode).strip()
        del past, logits, input_ids
        return {"text": text, "raw_text": raw_text, "tokens": len(generated),
                "generation_seconds": elapsed, "decoding": decoder.metadata(),
                "protection": protection,
                "first_token_seconds": first-began if first else None,
                "first_token_total_seconds": first-requested_at if first else None,
                "tokens_per_second": len(generated)/elapsed if elapsed else 0,
                "cancelled": stop_reason == "cancelled", "stop_reason": stop_reason}

