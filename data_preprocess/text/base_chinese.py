# -*- coding: utf-8 -*-
"""
流程：
1) 拉取 Chinese-Instruct 全量 => 100% 写出到 jsonl（text = prompt + 随机[，|：] + response）
2) 读取本地 <|im_start|>...<|im_end|> => 拆块后 100% 写出到 jsonl
3) 生成最终混合：仅 CI + IM（可选按 target_gb 提前截止）
4) 全程打印统计信息（条数/字节/还差多少等） + 英文字母阈值过滤

依赖：
pip install datasets regex tqdm
"""

import os, json, argparse, regex as re, random
from hashlib import md5
from tqdm import tqdm

try:
    from datasets import load_dataset, get_dataset_config_names
except Exception:
    load_dataset = None
    get_dataset_config_names = None

random.seed(42)

# ---------- 可调阈值 ----------
LETTER_LIMIT = 25   # 一条样本中 [A-Za-z] 超过这个数就丢弃

# ---------- 基础清洗 ----------
RE_CONTROL = re.compile(r"[\u200B-\u200F\uFEFF]")
RE_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")
RE_LATIN   = re.compile(r"[A-Za-z]")  # 仅统计 ASCII 字母

def clean_text(s: str) -> str:
    if not s: return ""
    s = RE_CONTROL.sub("", s)
    s = RE_EMOJI.sub("", s)
    s = s.replace("\u0000", "")
    return s.strip()

def too_many_letters(s: str, limit: int = LETTER_LIMIT) -> bool:
    """统计 ASCII 英文字母数量；超过 limit 判为英文/代码重样本，预训练阶段丢弃。"""
    return False

def byte_len(s: str) -> int:
    return len(s.encode("utf-8"))

def mk_record(text: str, qid: int) -> dict:
    text = clean_text(text)
    uk = md5(text.encode("utf-8")).hexdigest()
    return {"text": text, "question": qid, "uniqueKey": uk}

# ---------- 1) CI 全量 ----------
def list_ci_configs():
    ds_name = "Mxode/Chinese-Instruct"
    cfgs = []
    if get_dataset_config_names is not None:
        try:
            cfgs = get_dataset_config_names(ds_name)
        except Exception:
            cfgs = []
    return ds_name, (cfgs or [None])

def export_ci_all(out_path, start_qid=0, min_chars=1, max_chars=8000):
    assert load_dataset is not None, "请先 `pip install datasets`。"
    ds_name, cfgs = list_ci_configs()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    qid = start_qid
    total_bytes = 0
    total_rows  = 0
    total_dropped_letters = 0

    print(f"[CI] 将遍历以下子集：{cfgs}")
    with open(out_path, "w", encoding="utf-8") as w:
        for cfg in cfgs:
            ds = load_dataset(ds_name, cfg, split="train", streaming=True)
            cfg_rows = 0
            cfg_bytes= 0
            cfg_drop_letters = 0
            pbar = tqdm(ds, desc=f"CI/{cfg or 'default'}")
            for ex in pbar:
                prompt = ex.get("prompt") or ex.get("instruction") or ""
                resp   = ex.get("response") or ex.get("output") or ex.get("answer") or ""
                p = clean_text(prompt); r = clean_text(resp)
                if not p or not r:
                    continue
                # ===== 关键改动：prompt + 随机分隔符（逗号或冒号） + response =====
                sep = "，" if random.random() < 0.5 else "："
                txt = f"{p}{sep}{r}"
                if not (min_chars <= len(txt) <= max_chars):
                    continue
                if too_many_letters(txt):
                    cfg_drop_letters += 1
                    continue
                rec = mk_record(txt, qid)
                line = json.dumps(rec, ensure_ascii=False)
                w.write(line + "\n")
                qid += 1
                total_rows += 1
                total_bytes += byte_len(line) + 1
                cfg_rows  += 1
                cfg_bytes += byte_len(line) + 1
            total_dropped_letters += cfg_drop_letters
            print(f"[CI] 子集 {cfg or 'default'} 写出 {cfg_rows} 行，约 {cfg_bytes/1024**3:.3f} GB | 过滤(字母>{LETTER_LIMIT}) {cfg_drop_letters} 行")

    print(f"[CI] 总计写出 {total_rows} 行，约 {total_bytes/1024**3:.3f} GB 到 {out_path} | "
          f"总过滤(字母>{LETTER_LIMIT}) {total_dropped_letters} 行")
    return qid, total_rows, total_bytes  # 返回下一个 qid

# ---------- 2) 本地 IM 全量 ----------
RE_IM_BLOCK = re.compile(r"<\|im_start\|>(.*?)<\|im_end\|>", re.DOTALL)

def parse_im_blocks(s: str):
    return [clean_text(m.group(1)).strip() for m in RE_IM_BLOCK.finditer(s) if clean_text(m.group(1)).strip()]

def export_im_all(im_in, im_out, start_qid=0, min_chars=1, max_chars=8000):
    os.makedirs(os.path.dirname(im_out) or ".", exist_ok=True)
    qid = start_qid
    total_bytes = 0
    total_rows  = 0
    drop_letters = 0

    with open(im_out, "w", encoding="utf-8") as w, open(im_in, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="IM/clean"):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                raw = ex.get("text", "")
            except Exception:
                raw = line
            if not raw:
                continue
            blocks = parse_im_blocks(raw)
            for blk in blocks:
                if not (min_chars <= len(blk) <= max_chars):
                    continue
                if too_many_letters(blk):
                    drop_letters += 1
                    continue
                rec = mk_record(blk, qid)
                line_out = json.dumps(rec, ensure_ascii=False)
                w.write(line_out + "\n")
                qid += 1
                total_rows  += 1
                total_bytes += byte_len(line_out) + 1

    print(f"[IM] 总计写出 {total_rows} 行，约 {total_bytes/1024**3:.3f} GB 到 {im_out} | 过滤(字母>{LETTER_LIMIT}) {drop_letters} 行")
    return qid, total_rows, total_bytes

# ---------- 3) 仅 CI + IM 的最终混合 ----------
def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def build_final(ci_jsonl, im_jsonl, out_path, target_gb=None):
    """
    仅将 CI 与 IM 顺序写入 final。
    若 target_gb 指定，则在合并阶段达到体积上限后提前停止（不再写入更多样本）。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").close()  # truncate

    target_bytes = None if target_gb is None else int(target_gb * (1024 ** 3))
    wrote_rows = 0
    wrote_bytes = 0

    def append_source(src_path, tag, qid_start):
        nonlocal wrote_rows, wrote_bytes
        qid = qid_start
        rows = 0
        bytes_ = 0
        dropped_letters = 0

        with open(out_path, "a", encoding="utf-8") as w:
            for ex in tqdm(iter_jsonl(src_path), desc=f"APPEND/{tag}"):
                if target_bytes is not None and wrote_bytes >= target_bytes:
                    break
                txt = clean_text(ex.get("text", ""))
                if not txt:
                    continue
                if too_many_letters(txt):
                    dropped_letters += 1
                    continue
                rec = mk_record(txt, qid)
                line = json.dumps(rec, ensure_ascii=False)
                # 若有体积上限，最后一条也可能超出一点点——保持简单直接写入
                w.write(line + "\n")
                qid += 1
                rows   += 1
                bytes_ += byte_len(line) + 1
                wrote_rows  += 1
                wrote_bytes += byte_len(line) + 1

        print(f"[APPEND/{tag}] 写出 {rows} 行，约 {bytes_/1024**3:.3f} GB | 过滤(字母>{LETTER_LIMIT}) {dropped_letters} 行")
        return qid

    qid = 0
    qid = append_source(ci_jsonl, "CI", qid)
    qid = append_source(im_jsonl, "IM", qid)

    print(f"[FINAL] 最终合计 {wrote_rows} 行，约 {wrote_bytes/1024**3:.3f} GB 写入 {out_path}")
    if target_bytes is not None and wrote_bytes < target_bytes:
        remain = target_bytes - wrote_bytes
        print(f"[INFO] 目标 {target_gb:.3f} GB 未达到，当前约 {wrote_bytes/1024**3:.3f} GB，还差 ≈ {remain/1024**3:.3f} GB（仅 CI+IM，无额外补齐）。")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_gb", type=float, default=None, help="（可选）最终合并的体积上限，仅用于 CI+IM 阶段提前截止。")


    ci_out     = "./data/chinese_instruct_all.jsonl"
    im_in      = "../data/pretrain_hq.jsonl"
    im_out     = "./data/local_im_clean.jsonl"
    final_out  = "./data/pretrain_mix_ciim.jsonl"

    # 1) CI：全量导出（prompt + [，|：] + response）
    if not os.path.exists(ci_out):
        _, ci_rows, ci_bytes = export_ci_all(ci_out, start_qid=0)
    else:
        ci_bytes = sum(byte_len(line) + 1 for line in open(ci_out, "r", encoding="utf-8"))
        ci_rows  = sum(1 for _ in open(ci_out, "r", encoding="utf-8"))
        print(f"[CI] 已存在：{ci_rows} 行，约 {ci_bytes/1024**3:.3f} GB")

    # 2) 本地 IM：全量清洗导出
    if not os.path.exists(im_out):
        _, im_rows, im_bytes = export_im_all(im_in, im_out, start_qid=0)
    else:
        im_bytes = sum(byte_len(line) + 1 for line in open(im_out, "r", encoding="utf-8"))
        im_rows  = sum(1 for _ in open(im_out, "r", encoding="utf-8"))
        print(f"[IM] 已存在：{im_rows} 行，约 {im_bytes/1024**3:.3f} GB")

    # 3) 仅 CI + IM 合并（可选 target_gb）
    build_final(ci_out, im_out, final_out, target_gb=15)

if __name__ == "__main__":
    main()
