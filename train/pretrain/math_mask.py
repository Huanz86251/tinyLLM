import unicodedata
import torch


def _looks_like_markup(s: str) -> bool:
    # 粗略过滤 HTML/XML/自定义标签：含 <...> 就当作标记
    return ("<" in s) and (">" in s)


def _has_letter(s: str) -> bool:
    # 任何 Unicode 字母都算（含英文字母、希腊字母等）
    return any(unicodedata.category(ch).startswith("L") for ch in s)


def _strip_bpe_markers(s: str) -> str:
    # 处理类似 BPE 的 Ġ / ▁ 前缀
    return s.replace("Ġ", " ").replace("▁", " ")


def _is_single_core_op_token(raw_norm: str, surf_norm: str, op_chars) -> bool:
    """
    判断一个 token 是否是“单个核心运算符”：
    - 去掉空格 / BPE 标记后，只剩 1 个字符
    - 且该字符在 op_chars 内（比如 + - * / = ^ ＋－×÷＝％ 等）
    - 不包含任何数字（数字类交给 digit mask）
    """
    def normalize_body(s: str) -> str:
        # raw_norm / surf_norm 已经做过 NFKC，这里仅去掉空白
        s = s.strip()
        if not s:
            return ""
        # 防一手内部空白
        return "".join(ch for ch in s if not ch.isspace())

    for base in (raw_norm, surf_norm):
        body = normalize_body(base)
        if not body:
            continue
        # 里面如果带数字，就交给 digit，不当作纯运算符
        if any(unicodedata.category(ch) == "Nd" for ch in body):
            return False
        if len(body) == 1 and body in op_chars:
            return True
    return False


def build_mathish_masks(tokenizer):
    """
    按类别构造多种 mask:
    - digit: 含数字（半角 / 全角 / 其它 Nd）
    - math_op: “单个”加减乘除、幂、取模等核心运算符（过滤掉 //////、*** 这类噪声）
    - math_bracket: 各种括号（现在不再算进 is_mathish）
    - math_symbol: 典型数学符号（π、√、∑、∫、∞ 等；当前不再算进 is_mathish）
    - weak_punct: 常规句读标点（.,，。：:;；；当前不再算进 is_mathish）

    最终用于加权的 “数学-ish” 总 mask:
        is_mathish = digit | math_op
    也就是：只有“数字”和“核心运算符 token”才会被当成数学 token 提高权重。
    """

    vocab = tokenizer.get_vocab()
    V = len(vocab)
    id2tok = [None] * V
    for tok, tid in vocab.items():
        if tid < V:
            id2tok[tid] = tok

    # 分类用字符集合
    half_digits   = set("0123456789")
    full_digits   = set("０１２３４５６７８９")
    half_ops      = set("+-*/=^%")
    full_ops      = set("＋－×÷＝％")
    op_chars      = half_ops | full_ops
    brackets      = set("()（）[]【】{}")
    weak_punct    = set(".,，。：:;；")
    # strong_symbols 整体砍掉，不再单独加权
    # strong_symbols = set("‰∞π√∑∏∫≠≤≥≈")

    def scan_string_for_basic_flags(s: str, flags: dict):
        """只扫描数字 / 括号 / 弱标点，不再直接把任意含 +-*/ 视为 math_op。"""
        for ch in s:
            cat = unicodedata.category(ch)
            # 数字：ASCII / 全角数字 / 其它 Nd
            if ch in half_digits or ch in full_digits or cat == "Nd":
                flags["digit"] = True
            # 括号（只做统计用，不再进 is_mathish）
            if ch in brackets:
                flags["bracket"] = True
            # 弱标点（只做统计用，不再进 is_mathish）
            if ch in weak_punct:
                flags["weak_punct"] = True
            # 强数学符号：现在不再使用，直接忽略
            # if ch in strong_symbols:
            #     flags["symbol"] = True

    # 各类 mask
    is_digit        = torch.zeros(V, dtype=torch.bool)
    is_math_op      = torch.zeros(V, dtype=torch.bool)
    is_math_bracket = torch.zeros(V, dtype=torch.bool)
    is_math_symbol  = torch.zeros(V, dtype=torch.bool)
    is_weak_punct   = torch.zeros(V, dtype=torch.bool)

    for tid, raw_tok in enumerate(id2tok):
        if raw_tok is None:
            continue

        raw_norm = unicodedata.normalize("NFKC", _strip_bpe_markers(raw_tok))
        try:
            surf = tokenizer.decode(
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            surf = raw_norm
        surf_norm = unicodedata.normalize("NFKC", surf)

        # 跳过类似 <s> 之类标记 & 含字母的 token
        if _looks_like_markup(raw_norm) or _looks_like_markup(surf_norm):
            continue
        if _has_letter(raw_norm) or _has_letter(surf_norm):
            continue

        flags = {
            "digit": False,
            "op": False,       # 占位，不再直接用
            "bracket": False,
            "weak_punct": False,
            "symbol": False,   # 现在不会被置 True，只是占位
        }
        scan_string_for_basic_flags(raw_norm, flags)
        scan_string_for_basic_flags(surf_norm, flags)

        if flags["digit"]:
            is_digit[tid] = True
        if flags["bracket"]:
            is_math_bracket[tid] = True
        if flags["symbol"]:
            is_math_symbol[tid] = True
        if flags["weak_punct"]:
            is_weak_punct[tid] = True

        # ⭐ 运算符：只把“纯单字符核心运算符 token”视为 math_op
        # 例如 "+", "-", "*", "/", "×", "÷", "%", "^" 及其带空格/BPE 前缀的版本
        # 像 "////////", "***", "*/\\", "%);" 这类一律忽略
        if not flags["digit"]:
            if _is_single_core_op_token(raw_norm, surf_norm, op_chars):
                is_math_op[tid] = True

    masks = {
        "digit":        is_digit,
        "math_op":      is_math_op,
        "math_bracket": is_math_bracket,
        "math_symbol":  is_math_symbol,
        "weak_punct":   is_weak_punct,
    }

    # ⭐ 最终 “数学-ish” 只用数字 + 核心运算符，不再包含括号 / 弱标点 / 强符号
    masks["is_mathish"] = is_digit | is_math_op
    return masks


def build_per_id_weight_from_masks(
    masks,
    w_mathish: float = 1.2,   # 数字 / 核心 +-×÷^% 这类 token
    cap: float | None = 3.0,
    w_bracket: float = 1.05,  # 括号稍微给点权重就行
    w_weak_punct: float = 1.0 # 句号逗号这些默认不加权
):
    """
    依据不同子 mask 赋不同权重：
    - digit / math_op / math_symbol: w_mathish（比如 1.2）
    - math_bracket: w_bracket（比如 1.05）
    - weak_punct: w_weak_punct（默认 1.0，不加权）
    如果某个 token 同时属于多类，后赋值的类别会覆盖前面（强类别覆盖弱类别），避免权重爆炸。
    """
    # 兼容老代码：如果只有 is_mathish，就简单统一加权
    if "digit" not in masks and "math_op" not in masks and "math_symbol" not in masks:
        V = masks["is_mathish"].numel()
        w = torch.ones(V, dtype=torch.float32)
        w[masks["is_mathish"]] = float(w_mathish)
        if cap is not None:
            w.clamp_(max=float(cap))
        return w

    # 新逻辑
    V = next(iter(masks.values())).numel()
    w = torch.ones(V, dtype=torch.float32)

    # 先给弱类别，再给强类别（后者覆盖前者）
    if "math_bracket" in masks:
        w[masks["math_bracket"]] = float(w_bracket)

    if "weak_punct" in masks:
        w[masks["weak_punct"]] = float(w_weak_punct)

    # 强数学：数字、运算符、数学符号
    if "digit" in masks:
        w[masks["digit"]] = float(w_mathish)
    if "math_op" in masks:
        w[masks["math_op"]] = float(w_mathish)
    if "math_symbol" in masks:
        w[masks["math_symbol"]] = float(w_mathish)

    if cap is not None:
        w.clamp_(max=float(cap))
    return w


def preview_mathish_tokens(tokenizer, masks, k=20):
    """
    打印统计信息和前 k 个样例（按 token id 升序）。
    自动跳过特殊 token。
    """
    math_ids = torch.nonzero(masks["is_mathish"], as_tuple=False).view(-1).tolist()

    # 过滤特殊 token
    special_ids = set()
    try:
        for _, tok in tokenizer.special_tokens_map_extended.items():
            if tok is None:
                continue
            if isinstance(tok, (list, tuple)):
                for t in tok:
                    try:
                        special_ids.add(tokenizer.convert_tokens_to_ids(t))
                    except Exception:
                        pass
            else:
                try:
                    special_ids.add(tokenizer.convert_tokens_to_ids(tok))
                except Exception:
                    pass
    except Exception:
        pass

    math_ids = [tid for tid in math_ids if tid not in special_ids]

    vocab_size = len(tokenizer.get_vocab())
    ratio = len(math_ids) / max(1, vocab_size)

    sample_ids = sorted(math_ids)[:k]
    for tid in sample_ids:
        raw_tok = tokenizer.convert_ids_to_tokens(tid)
        try:
            surf = tokenizer.decode(
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            surf = raw_tok
        raw_norm  = unicodedata.normalize("NFKC", _strip_bpe_markers(raw_tok))
        surf_norm = unicodedata.normalize("NFKC", surf)

    return math_ids
