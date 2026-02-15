"""Conservative inference controls. Only generated history is penalized.
Benchmark evaluators retain their original decoding unless they explicitly opt in.
"""
from collections import Counter
import difflib, json, math, re, time
import torch
from project_paths import path

MATH_WORDS = {"frac","sqrt","boxed","left","right","begin","end","pm","sum","int","text",
              "cdot","quad","times","alpha","beta","theta","le","ge","sin","cos","log"}
def load_profile(mode):
    config = json.loads(path("configs/decoding.json").read_text(encoding="utf-8-sig"))
    p = {**config["common"], **config["modes"][mode]}
    if not 1 <= p["repetition_penalty"] <= 1.5:
        raise ValueError("repetition_penalty must be in [1, 1.5]")
    if p["strategy"] not in ("greedy", "sample"):
        raise ValueError("Unknown decoding strategy")
    if p["strategy"] == "sample" and not (0 < p["temperature"] <= 2 and 0 < p["top_p"] <= 1 and p["top_k"] >= 1
                                             and 0 <= p.get("min_p", 0) < 1):
        raise ValueError("Invalid sampling configuration")
    return p

def text_loop(text, repeats=3):
    """Exact repeated prose; symbolic math-only strings are deliberately exempt."""
    offset = max(0, len(text)-1800)
    tail = text[offset:]
    expression = re.compile(r"(?P<unit>.{16,180}?)(?P=unit){"+str(repeats-1)+r",}", re.S)
    for m in expression.finditer(tail):
        unit = m.group("unit")
        letters = re.findall(r"[a-zA-Z\u4e00-\u9fff]", unit)
        if len(letters) < 8 or len(set(letters)) < 3:
            continue
        end = offset+m.start()+len(unit)
        return {"kind": "exact_text_cycle", "keep_chars": end,
                "unit_chars": len(unit), "repeat_count": len(m.group())//len(unit)}
    return None


def low_information_tail(text, min_chars=64):
    """Detect a long numeric/punctuation-only suffix in prose modes.

    Tiny VLMs sometimes escape an ordinary sentence into years, percentages,
    hyphens and growing dot runs. This suffix carries no natural-language
    information and often evades exact n-gram checks because its punctuation
    count changes every step.
    """
    tail_offset=max(0,len(text)-640)
    tail=text[tail_offset:]
    allowed=r'''0-9０-９\s.,，。:;!?！？%％()\[\]{}+\-*/_=…·"“”'<>'''
    match=re.search(r"(["+allowed+r"]{"+str(min_chars)+r",})$",tail)
    if not match:
        return None
    noisy=match.group(1)
    compact=re.sub(r"\s+","",noisy)
    if not compact:
        return None
    punctuation=sum(not ch.isdigit() for ch in compact)/len(compact)
    if punctuation < 0.30 or len(set(compact)) > 24:
        return None
    keep=tail_offset+match.start(1)
    # If the noisy suffix follows a dangling clause, return the last complete
    # prose sentence instead of showing an awkward unfinished fragment.
    boundary=max(text.rfind(mark,0,keep) for mark in "。！？!?\n")
    if boundary >= 0 and keep-boundary <= 56:
        keep=boundary+1
    return {"kind":"low_information_tail","keep_chars":keep,
            "tail_chars":len(noisy),"unique_chars":len(set(compact))}


_HTML_BREAK = re.compile(r"<\s*(?:br\s*/?|/(?:p|div|h[1-6]|li|ul|ol))\s*>", re.I)
_HTML_TAG = re.compile(
    r"<\s*/?\s*(?:p|div|h[1-6]|span|section|article|header|footer|main|strong|b|em|i|ul|ol|li)"
    r"(?:\s+[^<>]*?)?\s*>",
    re.I,
)
_IMAGE_PLACEHOLDER = re.compile(r"<\s*img\s*/?\s*>", re.I)


def sanitize_generated_text(text, mode):
    """Remove training-data markup without touching mathematical comparison signs."""
    if mode not in ("chat", "vision"):
        return text
    cleaned = _IMAGE_PLACEHOLDER.sub("", text)
    cleaned = _HTML_BREAK.sub("\n", cleaned)
    cleaned = _HTML_TAG.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.lstrip()


_STUTTER_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "in", "on", "at", "to", "of", "and", "or", "with",
    "has", "have", "having", "looks", "appears", "image", "picture", "shows",
    "了", "的", "是", "在", "有", "和", "与", "这", "那", "图", "图片", "显示",
}


def semantic_stutter_tail(text):
    """Detect many short paraphrases about the same subject in prose output."""
    offset = max(0, len(text) - 2200)
    tail = text[offset:]
    boundary = re.compile(
        r"(.{3,260}?)(?:</(?:p|div|h[1-6]|li|ul|ol)>|<br\s*/?>|[.!?。！？；;]+|\n)",
        re.I | re.S,
    )
    clauses = []
    for match in boundary.finditer(tail):
        plain = sanitize_generated_text(match.group(1), "vision")
        words = re.findall(r"[a-zA-Z]{2,}|[\u4e00-\u9fff]{1,4}", plain.lower())
        terms = [word for word in words if word not in _STUTTER_STOP_WORDS]
        if len(terms) >= 2:
            clauses.append((set(terms), offset + match.end()))
    if len(clauses) < 8:
        return None
    recent = clauses[-8:]
    appearances = Counter(term for terms, _ in recent for term in terms)
    if not appearances:
        return None
    dominant, count = appearances.most_common(1)[0]
    if count < 6:
        return None
    all_terms = [term for terms, _ in recent for term in terms]
    if len(set(all_terms)) / max(1, len(all_terms)) > 0.58:
        return None
    occurrences = [end for terms, end in clauses if dominant in terms]
    keep = occurrences[min(2, len(occurrences) - 1)]
    return {
        "kind": "semantic_stutter",
        "keep_chars": keep,
        "dominant_term": dominant,
        "recent_clause_count": len(recent),
        "dominant_clause_count": count,
    }

def completed_boxed_answer(text):
    """Wait for a balanced boxed answer and its outer math delimiter."""
    start = text.rfind(r"\boxed")
    if start < 0:
        return False
    if "<|thought_start|>" in text and "<|thought_end|>" not in text:
        return False
    opening = text.find("{", start + 6)
    if opening < 0 or text[start + 6:opening].strip():
        return False
    depth, end = 1, opening + 1
    while end < len(text) and depth:
        if text[end] == "{" and text[end-1] != "\\":
            depth += 1
        elif text[end] == "}" and text[end-1] != "\\":
            depth -= 1
        end += 1
    if depth:
        return False
    # A completed box is often followed by $ or \). Do not clip the delimiter.
    normalized = re.sub(r"(?<!\\)\\\\([()[\]])", lambda m: "\\" + m[1], text)
    if len(re.findall(r"(?<!\\)\$", normalized)) % 2:
        return False
    if normalized.count(r"\(") > normalized.count(r"\)") or normalized.count(r"\[") > normalized.count(r"\]"):
        return False
    return True


def trim_history(text):
    found = text_loop(text, repeats=3)
    return (text[:found["keep_chars"]].rstrip(), found) if found else (text, None)

class SafeDecoder:
    def __init__(self, tokenizer, device, mode):
        self.tokenizer = tokenizer
        self.device = device
        self.mode = mode
        self.config = load_profile(mode)
        self.stop_ids = {int(x) for x in [tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>")] if x is not None and x >= 0}
        self.generator = torch.Generator(device=device).manual_seed(self.config["seed"])
        self.forbidden_ids = set()
        if mode in ("chat", "vision"):
            for literal in ("<", "</", ">"):
                encoded = tokenizer.encode(literal, add_special_tokens=False)
                if len(encoded) == 1:
                    self.forbidden_ids.add(int(encoded[0]))
                direct_id = tokenizer.convert_tokens_to_ids(literal)
                if direct_id is not None and direct_id >= 0:
                    self.forbidden_ids.add(int(direct_id))
            unknown = getattr(tokenizer, "unk_token_id", None)
            for literal in (
                "<img>", "<p>", "</p>", "<h1>", "</h1>", "<h2>", "</h2>",
                "<div>", "</div>", "<span>", "</span>", "<br>", "<br/>",
            ):
                token_id = tokenizer.convert_tokens_to_ids(literal)
                if token_id is not None and token_id != unknown and token_id >= 0:
                    self.forbidden_ids.add(int(token_id))
        self.stats = {"ngram_tokens_blocked": 0, "min_p_tokens_filtered": 0,
                      "penalty_steps": 0, "sampling_steps": 0,
                      "forbidden_markup_tokens": len(self.forbidden_ids)}
        self._math_cache = {}
        self._previous_inspect = -1

    def math_safe(self, token_id):
        if token_id not in self._math_cache:
            value = self.tokenizer.decode([token_id], skip_special_tokens=False).strip()
            self._math_cache[token_id] = (not value or bool(re.fullmatch(r"[\d\s+\-*/=^{}()\[\]$\\.,:;_<>|!%±√]+", value))
                                          or value in MATH_WORDS)
        return self._math_cache[token_id]

    def choose(self, logits, generated):
        scores = logits[0].float().clone()
        if torch.isnan(scores).any().item() or torch.isposinf(scores).any().item() or not torch.isfinite(scores).any().item():
            return None, {"kind": "invalid_logits"}
        raw_stop = {i: scores[i].item() for i in self.stop_ids if i < scores.numel()}
        if self.forbidden_ids:
            scores[list(self.forbidden_ids)] = -float("inf")
        counts = Counter(generated[-self.config["penalty_window"]:])
        penalized = [i for i in counts if i not in self.stop_ids and not self.math_safe(i)]
        if penalized:
            ix = torch.tensor(penalized, dtype=torch.long, device=scores.device)
            selected = scores[ix]
            penalty = self.config["repetition_penalty"]
            scores[ix] = torch.where(selected < 0, selected*penalty, selected/penalty)
            frequency = self.config["frequency_penalty"]
            if frequency:
                amounts = torch.tensor([min(counts[i],4) for i in penalized], dtype=scores.dtype, device=scores.device)
                scores[ix] -= frequency*amounts
            self.stats["penalty_steps"] += 1
        size = self.config["no_repeat_ngram_size"]
        history = generated[-self.config["ngram_window"]:]
        if size > 1 and len(history) >= size:
            prefix = history[-(size-1):]
            banned = set()
            for i in range(len(history)-size+1):
                if history[i:i+size-1] == prefix:
                    value = history[i+size-1]
                    if value not in self.stop_ids and not (self.mode in ("math","arc") and self.math_safe(value)):
                        banned.add(value)
            if banned:
                scores[list(banned)] = -float("inf")
                self.stats["ngram_tokens_blocked"] += len(banned)
        # Stop tokens are never reduced by repetition controls.
        for i, value in raw_stop.items():
            scores[i] = value
        if not torch.isfinite(scores).any().item():
            return None, {"kind": "no_valid_candidates"}
        if self.config["strategy"] == "greedy":
            return int(scores.argmax().item()), None
        scores /= self.config["temperature"]
        values, indices = torch.topk(scores, min(self.config["top_k"],scores.numel()))
        probabilities = torch.softmax(values, dim=-1)
        min_p = self.config.get("min_p", 0)
        if min_p:
            # Relative probability floor: confident steps become focused while
            # uncertain steps retain more alternatives. Always preserve argmax.
            min_remove = probabilities < probabilities.max()*min_p
            min_remove[probabilities.argmax()] = False
            self.stats["min_p_tokens_filtered"] += int(min_remove.sum().item())
            values[min_remove] = -float("inf")
            probabilities = torch.softmax(values, dim=-1)
        cumulative = probabilities.cumsum(-1)
        remove = cumulative-probabilities > self.config["top_p"]
        values[remove] = -float("inf")
        probabilities = torch.softmax(values, dim=-1)
        if not torch.isfinite(probabilities).all().item() or probabilities.sum().item() <= 0:
            return None, {"kind": "invalid_probabilities"}
        chosen = torch.multinomial(probabilities, 1, generator=self.generator)
        self.stats["sampling_steps"] += 1
        return int(indices[chosen].item()), None

    def inspect(self, text, generated):
        minimum = self.config["loop_min_tokens"]
        if len(generated) < minimum:
            return None
        # Four-token cadence bounds guard overhead; inspect punctuation immediately.
        if len(generated)%4 and (not text or text[-1] not in "，,。.!?！？;\n"):
            return None
        repeats = self.config["loop_repeats"]
        exact = text_loop(text, repeats)
        if exact:
            return exact
        if self.mode in ("chat","vision"):
            degraded = low_information_tail(text, self.config.get("degenerate_tail_min_chars",64))
            if degraded:
                return degraded
            stutter = semantic_stutter_tail(text)
            if stutter:
                return stutter
        # Catch long exact token cycles, including paragraphs and repeated subwords.
        for period in range(self.config["loop_min_period"], min(96, len(generated)//repeats)+1):
            unit = generated[-period:]
            if all(generated[-period*(j+1):-period*j] == unit for j in range(1,repeats)):
                start = len(generated)-period*repeats
                kept = self.tokenizer.decode(generated[:start+period], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
                text_unit = self.tokenizer.decode(unit, skip_special_tokens=True)
                letters = re.findall(r"[a-zA-Z\u4e00-\u9fff]",text_unit)
                if len(letters) >= 8 and len(set(letters)) >= 3:
                    return {"kind":"token_cycle","keep_chars":len(kept),"unit_tokens":period,"repeat_count":repeats}
        # Near-identical prose clauses; disabled for math/ARC to preserve derivations.
        if self.config["near_duplicate_clauses"]:
            matches = [m for m in re.finditer(r"[^，,。.!?！？;\n]+[，,。.!?！？;\n]", text[-1600:])
                       if len(m.group().strip()) >= 18]
            if len(matches) >= 3:
                recent = matches[-3:]
                norm = [re.sub(r"[\s，,。.!?！？;]+","",m.group()) for m in recent]
                ratios = [difflib.SequenceMatcher(None,norm[0],part,autojunk=False).ratio() for part in norm[1:]]
                if min(ratios) >= self.config["near_duplicate_ratio"]:
                    keep = max(0,len(text)-1600)+recent[0].end()
                    return {"kind":"near_duplicate_clauses","keep_chars":keep,"similarity":round(min(ratios),3)}
        return None

    def metadata(self):
        return {"config":dict(self.config),"stats":dict(self.stats)}

