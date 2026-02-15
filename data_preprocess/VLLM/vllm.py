from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import shutil
import time
import tempfile
from PIL import ImageOps

from PIL import UnidentifiedImageError
from datasets import load_dataset, Image as HFImage  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from tqdm import tqdm  # type: ignore

from PIL import Image  # type: ignore
from io import BytesIO
from PIL import Image
# =====================================================
# 配置区：根据你本地环境改这几项
# =====================================================
# ========== 分块保存配置 ==========
PARTS_ROOT = Path("vlm_sft_parts")  # 每个子集一个子目录（只放 jsonl + success）
PART_JSONL_NAME = "data.jsonl"
PART_SUCCESS_NAME = "success.txt"
MERGE_SUCCESS = PARTS_ROOT / "merge_success.txt"
TILE_SIZE = 448
GRID_SIZE = 2
CANVAS_SIZE = TILE_SIZE * GRID_SIZE  # 896
TOKENIZER_PATH = legacy_path(r"E:\learn\tiny_05B_cpt3\checkpoint-48000")  # 你的 tokenizer / ckpt 路径
OUTPUT_JSONL = "vlm_sft_mix_onevision_xkev_scienceqa.jsonl"  # 输出 JSONL
IMAGE_DIR = Path("vlm_sft_images")  # 输出图片文件夹（相对当前目录）
XKEV_IMAGE_ROOT = Path(r"D:\hf_datasets\Xkev-LLaVA-CoT-100k")
IMG_TOKEN = "<img>"  # vocab 里已有: "<img>": 7
SHUFFLE_BUFFER_DEFAULT = 20_000
SHUFFLE_BUFFER_BY_SUBSET = {
    "llava_instruct": 0,

    # 大集合：给 5k 够用，启动不会慢
    "tqa": 5_000,
    "dvqa": 5_000,
    "geo170k_qa": 5_000,
    "FigureQA": 5_000,
    "tabmwp": 5_000,

    # 小集合：别用 20k 默认值
    "ai2d": 2_000,
    "robut_sqa": 2_000,
    "chartqa": 2_000,
    "IconQA": 2_000,
    "Geometry3K": 2_000,
    "geomverse": 2_000,
    "intergps": 1_000,
    "mapqa": 2_000,
    "unigeo": 2_000,
}
# ================= FineVision 采样配比（总量≈963k，接近你之前≈945k） =================
FINEVISION_OCR_SUBSETS = {
    # OCR / 文本读图
    "ocrvqa": 90_000,
    "tal_ocr_eng": 90_000,
    "synthdog": 60_000,
    "docvqa": 8_000,
    "pdfvqa": 6_000,
    "textvqa": 18_000,
    "st_vqa": 15_000,
    "textocr(gpt4v)": 20_000,

    # 图表 / 表格读图（本质也是 OCR+读数）
    "chartqa": 18_000,
    "dvqa": 50_000,
    "figureqa": 25_000,
    "plotqa": 25_000,
    "tabmwp": 20_000,
    "Unichart": 30_000,
}

FINEVISION_GENERAL_SUBSETS = {
    # 通用“日常能力”：指令/对话/VQA（保持 35% 左右）
    "LLaVA_Instruct_150K": 70_000,
    "vision_flan(filtered)": 90_000,
    "visualwebinstruct(filtered)": 60_000,
    "vqav2": 50_000,
    "cocoqa": 25_000,
    "a_okvqa": 20_000,
    "sharegpt4v(llava)": 15_000,
}

FINEVISION_SCIENCE_FRIENDLY_SUBSETS = {
    # “看图推理”但不强调背知识：diagram/geo/map/geometry/icon
    "ai2d_merged": 5_000,
    "geo170k(align)": 35_000,
    "geo170k(qa)": 12_000,
    "mapqa": 30_000,
    "iconqa": 25_000,
    "geometry3k(mathv360k)": 9_000,
    "geomverse": 9_000,
    "intergps": 1_300,
    "unigeo(mathv360k)": 10_000,
    "geoqa+(mathv360k)": 12_000,

    # 少量 scienceqa（FineVision 内的版本），你说“不用太多”
    "scienceqa(nona_context)": 10_000,
}

FINEVISION_SUBSETS: dict[str, int] = {}
FINEVISION_SUBSETS.update(FINEVISION_OCR_SUBSETS)
FINEVISION_SUBSETS.update(FINEVISION_GENERAL_SUBSETS)
FINEVISION_SUBSETS.update(FINEVISION_SCIENCE_FRIENDLY_SUBSETS)

# streaming shuffle buffer：大集合给 5k/10k，小集合 1k/2k（启动快）
SHUFFLE_BUFFER_BY_SUBSET.update({
    # OCR 大头
    "ocrvqa": 10_000,
    "tal_ocr_eng": 10_000,
    "synthdog": 10_000,
    "Unichart": 10_000,
    "dvqa": 10_000,

    # 通用
    "vision_flan(filtered)": 10_000,
    "visualwebinstruct(filtered)": 10_000,
    "vqav2": 10_000,

    # 小集合
    "docvqa": 2_000,
    "pdfvqa": 2_000,
    "chartqa": 2_000,
    "figureqa": 2_000,
    "plotqa": 5_000,
    "tabmwp": 2_000,
    "textocr(gpt4v)": 2_000,
    "textvqa": 2_000,
    "st_vqa": 2_000,

    "ai2d_merged": 2_000,
    "geo170k(align)": 5_000,
    "geo170k(qa)": 5_000,
    "mapqa": 5_000,
    "iconqa": 2_000,
    "geometry3k(mathv360k)": 2_000,
    "geomverse": 2_000,
    "intergps": 1_000,
    "unigeo(mathv360k)": 2_000,
    "geoqa+(mathv360k)": 2_000,
    "scienceqa(nona_context)": 2_000,
})

# 全局 token 限制
MAX_TOTAL_TOKENS = 2048
MAX_THOUGHT_TOKENS = 500
RNG_SEED = 42
NUM_VSFT_EN   = 30_000
NUM_CAPTION_CN = 2_000
# -------- OneVision 子集采样配置 --------
# key = subset 名（对应 HF config），value = 采样条数上限
ONEVISION_SUBSETS = {
    # 注意：你代码里对 onevision/scienceqa 有 repeat=10 + max_samples=10_000_000 的特殊逻辑
    # 所以这里的数值几乎不生效（见你 process_onevision_subset 的 special-case）
    "scienceqa": 0,

    # --- 保一点通用指令（别让模型变“怪物”）---
    "llava_instruct": 55_000,   # 仍然建议不 shuffle（你已这么做）

    # --- 最贴 ScienceQA：教材/科学/图解 ---
    "tqa": 120_000,             # 实际会 cap 到 ~107k
    "ai2d": 30_000,             # 实际 cap 到 ~25.3k
    "robut_sqa": 15_000,        # 实际 cap 到 ~12.2k

    # --- 你之前缺的：图标/几何（对 ScienceQA 很加分）---
    "IconQA": 25_000,           # ~32.3k
    "Geometry3K": 20_000,       # ~23.7k
    "geomverse": 12_000,        # ~15.6k
    "intergps": 3_000,          # ~2.34k（写 3k 等价全吃）

    # --- 图表/数理（ScienceQA 读表/读图很关键）---
    "chartqa": 20_000,          # cap ~18.2k
    "dvqa": 95_000,             # 从 80k 上调（dvqa 总量很大）
    "FigureQA": 70_000,         # 从 50k 上调（FigureQA 总量 ~117k）
    "plotqa-part-00-of-10": 16_000,
    "plotqa-part-01-of-10": 16_000,

    # --- 表格文字题（ScienceQA 里遇到表格/数值推理会吃香）---
    "tabmwp": 25_000,           # 总量 ~56.1k

    # --- 地理/地图（ScienceQA 里也经常出现）---
    "GeoQA+": 30_000,           # cap ~26.8k
    "geo170k_qa": 90_000,       # 从 75k 上调（总量 ~134k）
    "mapqa": 50_000,            # 基本等于全吃（总量 ~52.6k）
    "unigeo": 20_000,           # cap ~18.8k
    "GEOS": 1_000,              # 总量 ~539

    # --- 轻量图文/阅读 ---
    "visualmrc": 10_000,         # cap ~4.94k
    "vistext": 20_000,           # cap ~15.4k
    "docvqa_train": 25_000,
    "textocr_gpt4v": 15_000,
}
# Xkev/LLaVA-CoT-100k：是否全量吃（通常建议 True）
USE_FULL_XKEV = True
NUM_XKEV = 100_000  # 如果 USE_FULL_XKEV=False，就按这个上限采样

# ScienceQA train：只取有 image + solution 的题目
NUM_SCIENCEQA_TRAIN = 30_000  # 上限，实际不会这么多

# 当存在 COT 时添加的英文 system 提示（和你前面文本 COT 脚本保持风格一致）
SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. "
    "Then provide the final answer."
)


# =====================================================
# 通用工具
# =====================================================
def _sanitize_part_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._+-]+", "_", name).strip("_")


def _part_paths(part_name: str) -> tuple[Path, Path, Path, Path]:
    """
    return: (part_dir, jsonl_final, jsonl_tmp, success_txt)
    """
    pn = _sanitize_part_name(part_name)
    part_dir = PARTS_ROOT / pn
    jsonl_final = part_dir / PART_JSONL_NAME
    jsonl_tmp = part_dir / (PART_JSONL_NAME + ".tmp")
    success_txt = part_dir / PART_SUCCESS_NAME
    return part_dir, jsonl_final, jsonl_tmp, success_txt


def _atomic_replace(src: Path, dst: Path) -> None:
    os.replace(str(src), str(dst))


def _read_success(success_txt: Path) -> dict | None:
    if not success_txt.exists():
        return None
    try:
        s = success_txt.read_text(encoding="utf-8").strip()
        if not s:
            return None
        return json.loads(s)
    except Exception:
        return None
def shuffle_jsonl_buckets(
    in_jsonl: Path,
    out_jsonl: Path,
    seed: int = 42,
    num_buckets: int = 256,
) -> None:
    """
    近似全局 shuffle，但不需要把全文件读进内存：
    1) 先按随机把每行分流到 num_buckets 个临时桶文件
    2) 再逐桶读入内存 shuffle 后写回输出
    """
    rnd = random.Random(seed)
    out_tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    if out_tmp.exists():
        out_tmp.unlink()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bucket_paths = [td / f"bucket_{i:04d}.jsonl" for i in range(num_buckets)]
        fps = [p.open("w", encoding="utf-8") for p in bucket_paths]

        # 1) 分桶
        with in_jsonl.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                b = rnd.randrange(num_buckets)
                fps[b].write(line if line.endswith("\n") else (line + "\n"))

        for f in fps:
            f.close()

        # 2) 桶内 shuffle + 写出
        with out_tmp.open("w", encoding="utf-8") as fout:
            for p in bucket_paths:
                if not p.exists() or p.stat().st_size == 0:
                    continue
                lines = p.read_text(encoding="utf-8").splitlines(True)  # 保留换行
                rnd.shuffle(lines)
                fout.writelines(lines)

    os.replace(str(out_tmp), str(out_jsonl))
    print(f">>> [SHUFFLE DONE] out={out_jsonl}")

def _write_success(success_txt: Path, payload: dict) -> None:
    success_txt.parent.mkdir(parents=True, exist_ok=True)
    success_txt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_partial_part(part_name: str, image_dir: Path) -> None:
    """
    如果某个 part 之前跑崩了：
    - 尝试从它的 jsonl/jsonl.tmp 里读出已经写出的 image_file
    - 把这些半截图片从全局 image_dir 删掉
    - 再把这个 part_dir 整体删掉，保证重跑干净
    """
    part_dir, jsonl_final, jsonl_tmp, success_txt = _part_paths(part_name)

    # 已经成功就不动
    ok = _read_success(success_txt)
    if ok and ok.get("ok") is True:
        return

    img_files: set[str] = set()

    def _collect_from(path: Path) -> None:
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        fn = rec.get("image_file", None)
                        if isinstance(fn, str) and fn:
                            img_files.add(fn)
                        elif isinstance(fn, list):
                            for x in fn:
                                if isinstance(x, str) and x:
                                    img_files.add(x)
                    except Exception:
                        # 坏行就忽略
                        pass
        except Exception:
            pass

    _collect_from(jsonl_tmp)
    _collect_from(jsonl_final)

    # 删掉已产生的半截图片（只删这个 part 自己 jsonl 里引用过的）
    for fn in img_files:
        p = image_dir / fn
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # 删除 part_dir
    if part_dir.exists():
        shutil.rmtree(part_dir, ignore_errors=True)


def run_part(
    part_name: str,
    runner_fn,
    tok: AutoTokenizer,
    image_dir: Path,
    start_index: int,
) -> tuple[int, int, int, bool]:
    """
    runner_fn: (tok, fout, image_dir, start_index) -> (next_index, kept, total_bytes)

    return: (next_index, kept, total_bytes, skipped)
    """
    PARTS_ROOT.mkdir(parents=True, exist_ok=True)
    part_dir, jsonl_final, jsonl_tmp, success_txt = _part_paths(part_name)

    ok = _read_success(success_txt)
    if ok and ok.get("ok") is True and jsonl_final.exists():
        next_index = int(ok["next_index"])
        kept = int(ok.get("kept", 0))
        total_bytes = int(ok.get("bytes", 0))
        print(f">>> [SKIP] {part_name} done. next_index={next_index}, kept={kept}")
        return next_index, kept, total_bytes, True

    # 没成功：先清理半截
    _cleanup_partial_part(part_name, image_dir)

    # 重新创建目录
    part_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kept = 0
    total_bytes = 0
    next_index = start_index

    with jsonl_tmp.open("w", encoding="utf-8") as fout:
        next_index, kept, total_bytes = runner_fn(tok, fout, image_dir, start_index)
        fout.flush()

    _atomic_replace(jsonl_tmp, jsonl_final)

    payload = {
        "ok": True,
        "part": part_name,
        "start_index": start_index,
        "next_index": next_index,
        "kept": kept,
        "bytes": total_bytes,
        "seconds": round(time.time() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_success(success_txt, payload)
    print(f">>> [DONE] {part_name}: start={start_index}, next={next_index}, kept={kept}")
    return next_index, kept, total_bytes, False


def merge_parts_jsonl(part_names: list[str], out_jsonl: Path, image_dir: Path) -> None:
    """
    只合并 jsonl（图片已经是全局 index 命名，早就在 image_dir 里了）
    """
    if MERGE_SUCCESS.exists() and out_jsonl.exists():
        print(">>> [SKIP] merge already done.")
        return

    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    total_lines = 0
    total_bytes = 0

    with tmp.open("w", encoding="utf-8") as fout:
        for part_name in part_names:
            part_dir, jsonl_final, _, success_txt = _part_paths(part_name)
            ok = _read_success(success_txt)
            if not (ok and ok.get("ok") is True and jsonl_final.exists()):
                raise RuntimeError(f"part not finished, cannot merge: {part_name}")

            with jsonl_final.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    # 可选：校验图片存在（merge 过程本来就要读一遍 jsonl，顺手做一致性检查）
                    try:
                        rec = json.loads(line)
                        fn = rec.get("image_file")
                        files = []
                        if isinstance(fn, str) and fn:
                            files = [fn]
                        elif isinstance(fn, list):
                            files = [x for x in fn if isinstance(x, str) and x]

                        for f in files:
                            if not (image_dir / f).exists():
                                raise FileNotFoundError(f"missing image for merge: {f} (part={part_name})")
                    except Exception as e:
                        raise RuntimeError(f"bad record while merge (part={part_name}): {e}")

                    fout.write(line if line.endswith("\n") else (line + "\n"))
                    total_lines += 1
                    total_bytes += len(line.encode("utf-8"))

    _atomic_replace(tmp, out_jsonl)
    MERGE_SUCCESS.write_text(
        json.dumps(
            {"ok": True, "samples": total_lines, "bytes": total_bytes, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f">>> [MERGE DONE] samples={total_lines}, out={out_jsonl}")

def apply_template(tok: AutoTokenizer, messages: List[Dict[str, Any]]) -> Tuple[str, int]:
    """
    用你自己的 chat_template 渲染，并数 token。
    不添加 system（由 messages 自己决定要不要）。
    """
    txt = tok.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    return txt, len(ids)
from io import BytesIO
from PIL import Image

import numpy as np

def resize_with_padding(img, target=448):
    w, h = img.size
    scale = min(target / w, target / h)   # 把长边压进 448，或短边放大到 448，看你喜好
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # 新建 448×448 背景（黑或白都行，看你训练习惯）
    PAD_RGB = (124, 116, 104)  # 按 image_mean * 255 取整

    canvas = Image.new("RGB", (target, target), PAD_RGB)
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))
    return canvas

def save_image_generic(img_field: Any, out_path: Path) -> bool:
    """
    把 HF 样本里的 image 字段保存到 out_path（统一存成 JPEG）。

    兼容几种常见形态：
    - PIL.Image.Image
    - str 路径
    - dict{"bytes": ... , "path": ...}  （Image(decode=False) 常见）
    """
    try:
        # 1) 已经是 PIL.Image
        if isinstance(img_field, Image.Image):
            img = img_field

        # 2) 直接是字符串路径
        elif isinstance(img_field, str):
            img = Image.open(img_field)

        # 3) datasets.Image(decode=False) 一般是 dict
        elif isinstance(img_field, dict):
            raw_bytes = img_field.get("bytes", None)
            path = img_field.get("path", None)
            img = None

            # ✅ 优先用 bytes（最稳），失败再尝试 path
            if raw_bytes is not None:
                try:
                    img = Image.open(BytesIO(raw_bytes))
                except Exception:
                    img = None

            if img is None and path:
                try:
                    img = Image.open(path)
                except Exception:
                    return False  # bytes 和 path 都不行，判定为坏图

            if img is None:
                return False

        else:
            # 其余乱七八糟的类型全部当失败处理
            return False

        if img.mode != "RGB":
            img = img.convert("RGB")
        #img = resize_with_padding(img, target=448)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG")
        return True

    except Exception:
        # 任何解码/保存错误都当作坏图
        return False

def _decode_to_pil(img_field: Any) -> Image.Image:
    # 复用你 save_image_generic 的兼容思路（PIL / str / dict{bytes,path}）
    if isinstance(img_field, Image.Image):
        img = img_field
    elif isinstance(img_field, str):
        img = Image.open(img_field)
    elif isinstance(img_field, dict):
        raw_bytes = img_field.get("bytes", None)
        path = img_field.get("path", None)
        if raw_bytes is not None:
            img = Image.open(BytesIO(raw_bytes))
        elif path is not None:
            img = Image.open(path)
        else:
            raise ValueError("bad image dict: no bytes/path")
    else:
        raise ValueError(f"unsupported image field type: {type(img_field)}")

    # 处理 EXIF 旋转，避免有些图片方向不对
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
def resize_with_padding_any(
    img: Image.Image,
    target: int,
    pad_rgb: tuple[int, int, int] = (124, 116, 104),
) -> tuple[Image.Image, dict]:
    """
    返回:
      - canvas: (target, target) RGB
      - meta: 记录 resize+pad 的几何参数（后续可反解 patch->原图坐标、算 padding mask）
    """
    w, h = img.size
    scale = min(target / w, target / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target, target), pad_rgb)
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    meta = {
        "src_size": [w, h],
        "target": target,
        "scale": float(scale),
        "new_size": [new_w, new_h],
        "offset": [int(offset_x), int(offset_y)],
        # content_box 是 canvas 上“非 padding”的矩形区域
        "content_box": [int(offset_x), int(offset_y), int(offset_x + new_w), int(offset_y + new_h)],
        "pad_rgb": [int(pad_rgb[0]), int(pad_rgb[1]), int(pad_rgb[2])],
    }
    return canvas, meta
def save_image_2x2_plus_thumb(
    img_field: Any,
    image_dir: Path,
    index: int,
    jpeg_quality: int = 100,
) -> tuple[list[str], dict] | None:
    jpeg_quality = int(jpeg_quality)
    if not 80 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality must be in [80, 100], got {jpeg_quality}")
    out_names = [f"{index}-1.jpg", f"{index}-2.jpg", f"{index}-3.jpg", f"{index}-4.jpg", f"{index}-5.jpg"]
    out_paths = [image_dir / n for n in out_names]

    UPSCALE_MIN_EDGE = 528
    SMALL_EDGE_TH = 800
    OV_SMALL = 0.2
    OV_BIG = 0.1

    saved: list[Path] = []
    try:
        img0 = _decode_to_pil(img_field)
        orig_w, orig_h = img0.size

        # -------- 1) 可选上采样（你原逻辑不动） --------
        img = img0
        w, h = img.size
        work_scale = 1.0
        min_side = min(w, h)
        if min_side < TILE_SIZE:
            denom = max(1, min_side)
            scale = UPSCALE_MIN_EDGE / denom
            new_w = max(2, int(round(w * scale)))
            new_h = max(2, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size
            work_scale = w / orig_w  # 记录从 orig -> work 的缩放（等比）

        # -------- 2) 动态 overlap（你原逻辑不动） --------
        min_side = min(w, h)
        ov_ratio = OV_SMALL if min_side < SMALL_EDGE_TH else OV_BIG

        mx, my = w // 2, h // 2
        ov = int(round(ov_ratio * min_side))

        ov_cap = min(mx, my) - 1
        if ov_cap < 0:
            ov = 0
        else:
            ov = max(0, min(ov, ov_cap))

        # -------- 3) 2x2 boxes（补全你文件里被省略的部分） --------
        boxes = [
            (0,       0,       mx + ov, my + ov),   # 1 左上
            (mx - ov, 0,       w,       my + ov),   # 2 右上
            (0,       my - ov, mx + ov, h),         # 3 左下
            (mx - ov, my - ov, w,       h),         # 4 右下
        ]

        # -------- 4) 做 tiles + meta --------
        tiles: list[Image.Image] = []
        view_metas: list[dict] = []

        for view_idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
            if x1 - x0 < 2 or y1 - y0 < 2:
                return None

            crop = img.crop((x0, y0, x1, y1))
            tile448, pad_meta = resize_with_padding_any(crop, target=TILE_SIZE)

            tiles.append(tile448)
            view_metas.append({
                "view_idx": view_idx,              # 1..4
                "type": "tile",
                "file": out_names[view_idx - 1],
                "crop_box_work": [int(x0), int(y0), int(x1), int(y1)],  # 在 work 图上的 crop
                "crop_size_work": [int(x1 - x0), int(y1 - y0)],
                "pad": pad_meta,                   # resize+pad 参数（可反解映射/算mask）
                "grid_pos": [int((view_idx - 1) // 2), int((view_idx - 1) % 2)],  # (row,col) in 2x2
            })

        # thumb（用 work 图生成，保持你原行为）
        thumb448, thumb_pad_meta = resize_with_padding_any(img, target=TILE_SIZE)

        view_metas.append({
            "view_idx": 5,
            "type": "thumb",
            "file": out_names[4],
            "crop_box_work": [0, 0, int(w), int(h)],
            "crop_size_work": [int(w), int(h)],
            "pad": thumb_pad_meta,
            "grid_pos": None,
        })

        # -------- 5) 写盘（不变） --------
        for t, p in zip(tiles, out_paths[:4]):
            p.parent.mkdir(parents=True, exist_ok=True)
            t.save(p, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
            saved.append(p)

        out_paths[4].parent.mkdir(parents=True, exist_ok=True)
        thumb448.save(out_paths[4], format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
        saved.append(out_paths[4])

        # -------- 6) 打包 vision_meta（这是你后面 Q-Former/映射要用的） --------
        # InternViT-448: patch_size=14 => 32x32=1024（不含 cls）
        vision_meta = {
            "vision_encoder": {
                "name": "OpenGVLab/InternViT-300M-448px-V2_5",
                "image_size": 448,
                "patch_size": 14,
                "grid_hw": [32, 32],
                "num_patches": 1024,
                # 如果你后面走 InternVL 那套 pixel-unshuffle，会变成 256 tokens；这里先明确记录
                "note": "raw InternViT patches are 32x32=1024 (plus optional CLS). InternVL-style pixel-unshuffle would reduce tokens to 256.",
            },
            "source": {
                "orig_size": [int(orig_w), int(orig_h)],
                "work_size": [int(w), int(h)],
                "work_scale": float(work_scale),   # orig -> work
            },
            "tiling": {
                "scheme": "2x2_plus_thumb",
                "overlap_ratio": float(ov_ratio),
                "overlap_pixels_work": int(ov),
                "mx_my_work": [int(mx), int(my)],
            },
            "views": view_metas,
        }

        return out_names, vision_meta

    except Exception:
        for p in saved:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        return None


# ========== COT 专用小工具（沿用你文本 COT 脚本的风格） ==========

def build_messages_with_cot(
    user_text: str,
    answer: str,
    thought: str | None,
    use_system: bool = True,
) -> List[Dict[str, Any]] | None:
    """
    - 有 thought 且 use_system=True: 在最前添加 system，再用 'thought' 字段
    - 有 thought 且 use_system=False: 仅 assistant 带 'thought'
    - 无 thought: 普通 assistant 回答（不带 'thought'）
    """
    user_text = (user_text or "").strip()
    answer = (answer or "").strip()
    if not user_text or not answer:
        return None

    messages: List[Dict[str, Any]] = []
    if use_system:
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT_FOR_COT,
        })

    messages.append({"role": "user", "content": user_text})

    if thought is not None and thought.strip():
        messages.append({
            "role": "assistant",
            "thought": thought.strip(),
            "content": answer,
        })
    else:
        messages.append({
            "role": "assistant",
            "content": answer,
        })
    return messages


def count_thought_tokens(tok: AutoTokenizer, thought: str) -> int:
    """
    只统计 <|thought_start|>thought<|thought_end|> 这一段的 token 数。
    """
    seg = "<|thought_start|>\n" + thought + "\n<|thought_end|>\n"
    ids = tok(seg, add_special_tokens=False)["input_ids"]
    return len(ids)


# =====================================================
# 1) OneVision 子集：mvp-lab/LLaVA-OneVision-1.5-Instruct-Data
#     字段: id, image(Image), conversations(list[{role, content}]), data_source
# =====================================================
def process_onevision_subset(
    tok: AutoTokenizer,
    fout,
    image_dir: Path,
    subset_name: str,
    max_samples: int,
    start_index: int,
    repeat: int = 1,
) -> Tuple[int, int, int]:
    """
    改成 streaming：不会为了 shuffle/len 去把整个 split 全量下载。
    - 对超大子集（llava_instruct / tqa / ...）会非常关键
    - 小子集也能用（scienceqa 你 repeat=3 也 OK，会重复走 3 轮）
    """
    print(f">>> 加载 OneVision 子集: {subset_name} (streaming=True) ...")

    # ✅ 关键：streaming=True
    ds = load_dataset(
        "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data",
        subset_name,
        split="train",
        streaming=True,
    )

    kept = 0
    total_bytes = 0
    cur_index = start_index

    dropped_no_image = 0
    dropped_bad_conv = 0
    dropped_too_long = 0
    dropped_save_fail = 0
    dropped_decode_error = 0

    # streaming 下没有“真实 orig_idx”，用 seen 计数代替（仅用于记录）
    seen = 0
    for epoch in range(repeat):
        # ✅ buffer shuffle：近似随机，但不会全量 materialize
        buf = SHUFFLE_BUFFER_BY_SUBSET.get(subset_name, SHUFFLE_BUFFER_DEFAULT)
        if buf and buf > 1:
            ds_epoch = ds.shuffle(buffer_size=buf, seed=RNG_SEED + epoch)
        else:
            ds_epoch = ds  # 不shuffle

        pbar = tqdm(
            total=None if max_samples >= 10_000_000 else max_samples,
            desc=f"OneVision/{subset_name} (ep{epoch + 1})",
        )


        it = iter(ds_epoch)

        while True:
            if kept >= max_samples:
                break

            try:
                ex = next(it)
            except StopIteration:
                break
            except (UnidentifiedImageError, OSError, ValueError) as e:
                # ✅ 这里就是你这次崩溃的位置：坏图/坏bytes/截断图等
                dropped_decode_error += 1
                continue
            except Exception:
                # 保险：任何上游 decode/iterable 异常都当作坏样本跳过
                dropped_decode_error += 1
                continue

            seen += 1
            orig_idx = seen - 1


            try:
                img_field = ex.get("image", None)
            except Exception:
                dropped_decode_error += 1
                continue

            if img_field is None:
                dropped_no_image += 1
                continue

            conv = ex.get("conversations", None)
            if not conv or len(conv) == 0:
                dropped_bad_conv += 1
                continue

            messages: List[Dict[str, Any]] = []
            for turn in conv:
                role_raw = turn.get("role") or turn.get("from") or ""
                role = "assistant" if role_raw in ("assistant", "gpt") else "user"
                content = (turn.get("content") or turn.get("value") or "").replace("<image>", IMG_TOKEN).strip()
                if not content:
                    continue
                messages.append({"role": role, "content": content})

            if len(messages) < 2:
                dropped_bad_conv += 1
                continue

            try:
                txt, tok_len = apply_template(tok, messages)
            except Exception:
                dropped_bad_conv += 1
                continue

            if tok_len > MAX_TOTAL_TOKENS:
                dropped_too_long += 1
                continue

            ret = save_image_2x2_plus_thumb(img_field, image_dir, cur_index)
            if ret is None:
                dropped_save_fail += 1
                continue

            img_files, vision_meta = ret

            rec = {
                "index": cur_index,
                "TXT": txt,
                "dataset": f"mvp-lab/LLaVA-OneVision-1.5-Instruct-Data/{subset_name}",
                "source_index": orig_idx,
                "image_file": img_files,   # <- list[str]
                "length": tok_len,
                "vision_meta": vision_meta,
                "lang": "en",
            }

            line = json.dumps(rec, ensure_ascii=False) + "\n"
            fout.write(line)
            total_bytes += len(line.encode("utf-8"))

            kept += 1
            cur_index += 1

            # 进度条按“成功保留的样本”更新，直观
            pbar.update(1)

        pbar.close()

        if kept >= max_samples:
            break

    print(
        f"  [OneVision/{subset_name}] kept={kept}, "
        f"dropped_no_image={dropped_no_image}, "
        f"dropped_bad_conv={dropped_bad_conv}, "
        f"dropped_too_long={dropped_too_long}, "
        f"dropped_save_fail={dropped_save_fail}, "
        f"dropped_decode_error={dropped_decode_error}"
    )
    return cur_index, kept, total_bytes


# =====================================================
# 2) Xkev/LLaVA-CoT-100k （视觉 COT，含 ScienceQA 等）
#     字段: id, image(Image), conversations(list[{from,value}])
#     gpt 段落用 <SUMMARY> / <CAPTION> / <REASONING> / <CONCLUSION>
# =====================================================

def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    if not m:
        return None
    return m.group(1).strip()

def _xkev_resolve_image_path(rel: str, roots: list[Path]) -> str:
    rel = rel.lstrip("/\\")
    for r in roots:
        p = r / rel
        if p.exists():
            return str(p)
    # 兜底：返回第一个 root 下的拼接（方便你打印排查）
    return str(roots[0] / rel)
def process_xkev_llava_cot(tok, fout, image_dir: Path, max_samples: int, start_index: int):
    print(">>> 加载 Xkev/LLaVA-CoT-100k (LOCAL train.jsonl)...")

    # 你这个目录里应该同时有 train.jsonl + 解压出来的 pisc/... 等文件夹
    local_jsonl = XKEV_IMAGE_ROOT / "train.jsonl"
    ds = load_dataset("json", data_files=str(local_jsonl), split="train")

    # Xkev 图片的可能根目录候选：按你 unzip 的落盘结构兜底
    IMG_ROOTS = [
        XKEV_IMAGE_ROOT,
        XKEV_IMAGE_ROOT / "image",
        XKEV_IMAGE_ROOT / "images",
    ]

    indices = list(range(len(ds)))
    random.shuffle(indices)

    kept = 0
    total_bytes = 0
    cur_index = start_index

    dropped_save_fail = 0
    dropped_no_image = 0
    dropped_bad_conv = 0
    dropped_decode_error = 0
    dropped_no_thought = 0
    dropped_thought_too_long = 0
    dropped_total_too_long = 0

    for orig_idx in tqdm(indices, desc="LLaVA-CoT-100k"):
        if kept >= max_samples:
            break

        try:
            ex = ds[orig_idx]
        except Exception:
            dropped_decode_error += 1
            continue

        img_rel = ex.get("image", None)
        if not img_rel:
            dropped_no_image += 1
            continue

        # ✅ 关键：把 "pisc/image/24231.jpg" 变成你本地真实存在的绝对路径
        img_abs = _xkev_resolve_image_path(img_rel, IMG_ROOTS)

        conv = ex.get("conversations", None)
        if not conv or len(conv) < 2:
            dropped_bad_conv += 1
            continue

        human_msg = next((c for c in conv if c.get("from") in ("human", "user")), None)
        gpt_msg = next((c for c in conv if c.get("from") in ("gpt", "assistant")), None)
        if human_msg is None or gpt_msg is None:
            dropped_bad_conv += 1
            continue

        user_raw = human_msg.get("value", "") or human_msg.get("content", "")
        asst_raw = gpt_msg.get("value", "") or gpt_msg.get("content", "")

        if not user_raw or not asst_raw:
            dropped_bad_conv += 1
            continue

        user_text = user_raw.replace("<image>", IMG_TOKEN).strip()

        summary = extract_tag(asst_raw, "SUMMARY")
        caption = extract_tag(asst_raw, "CAPTION")
        reasoning = extract_tag(asst_raw, "REASONING")
        conclusion = extract_tag(asst_raw, "CONCLUSION")

        thought_parts = [p for p in (summary, caption, reasoning) if p]
        thought = "\n\n".join(thought_parts) if thought_parts else None
        answer = conclusion or asst_raw.strip()

        if thought is None or not thought.strip():
            dropped_no_thought += 1
            continue

        try:
            thought_len = count_thought_tokens(tok, thought)
        except Exception:
            dropped_bad_conv += 1
            continue

        if thought_len > MAX_THOUGHT_TOKENS:
            dropped_thought_too_long += 1
            continue

        messages = build_messages_with_cot(
            user_text=user_text,
            answer=answer,
            thought=thought,
            use_system=True,
        )
        if messages is None:
            dropped_bad_conv += 1
            continue

        try:
            txt, total_len = apply_template(tok, messages)
        except Exception:
            dropped_bad_conv += 1
            continue

        if total_len > MAX_TOTAL_TOKENS:
            dropped_total_too_long += 1
            continue

        ret = save_image_2x2_plus_thumb(img_abs, image_dir, cur_index)
        if ret is None:
            dropped_save_fail += 1
            # ✅ 强烈建议：第一次失败就把路径打印出来，你立刻就知道是不是没解压到位
            # print("MISS:", img_rel, "=>", img_abs)
            continue

        img_files, vision_meta = ret
        rec = {
            "index": cur_index,
            "TXT": txt,
            "dataset": "Xkev/LLaVA-CoT-100k",
            "source_index": orig_idx,
            "image_file": img_files,
            "length": total_len,
            "lang": "en",
            "has_cot": True,
            "vision_meta": vision_meta,
            "cot_source": "llava_cot_100k",
        }

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        fout.write(line)
        total_bytes += len(line.encode("utf-8"))

        kept += 1
        cur_index += 1

    print(
        f"  [LLaVA-CoT-100k] kept={kept}, "
        f"dropped_no_image={dropped_no_image}, "
        f"dropped_bad_conv={dropped_bad_conv}, "
        f"dropped_no_thought={dropped_no_thought}, "
        f"dropped_thought_too_long={dropped_thought_too_long}, "
        f"dropped_total_too_long={dropped_total_too_long}, "
        f"dropped_save_fail={dropped_save_fail}",
        f"dropped_decode_error={dropped_decode_error}"
    )
    return cur_index, kept, total_bytes


# =====================================================
# 3) derek-thomas/ScienceQA: 只用 train split 里有 image + solution 的题
#     字段: id, image(Image), question, choices(list[str]), answer(int),
#           lecture(str), hint(str), solution(str)
# =====================================================

SCIENCEQA_LETTERS = ["A", "B", "C", "D", "E", "F"]
def _norm_img_token(s: str) -> str:
    # FineVision 里可能有不同占位写法，统一替换成你词表里的 <img>
    s = s.replace("<image>", IMG_TOKEN)
    s = re.sub(r"<image_\d+>", IMG_TOKEN, s)
    s = s.replace("[image]", IMG_TOKEN)
    s = s.replace("<img>", IMG_TOKEN)
    return s

def process_finevision_subset(
    tok: AutoTokenizer,
    fout,
    image_dir: Path,
    subset_name: str,
    max_samples: int,
    start_index: int,
    repeat: int = 1,
) -> Tuple[int, int, int]:
    print(f">>> 加载 FineVision 子集: {subset_name} (streaming=True) ...")

    ds = load_dataset(
        "HuggingFaceM4/FineVision",
        subset_name,
        split="train",
        streaming=True,
    )

    kept = 0
    total_bytes = 0
    cur_index = start_index

    dropped_no_image = 0
    dropped_multi_image = 0
    dropped_bad_texts = 0
    dropped_too_long = 0
    dropped_save_fail = 0
    dropped_decode_error = 0

    seen = 0
    for epoch in range(repeat):
        buf = SHUFFLE_BUFFER_BY_SUBSET.get(subset_name, SHUFFLE_BUFFER_DEFAULT)
        if buf and buf > 1:
            ds_epoch = ds.shuffle(buffer_size=buf, seed=RNG_SEED + epoch)
        else:
            ds_epoch = ds

        pbar = tqdm(
            total=None if max_samples >= 10_000_000 else max_samples,
            desc=f"FineVision/{subset_name} (ep{epoch + 1})",
        )

        it = iter(ds_epoch)
        while True:
            if kept >= max_samples:
                break

            try:
                ex = next(it)
            except StopIteration:
                break
            except (UnidentifiedImageError, OSError, ValueError):
                dropped_decode_error += 1
                continue
            except Exception:
                dropped_decode_error += 1
                continue

            seen += 1
            orig_idx = seen - 1

            images = ex.get("images", None)
            if not images or not isinstance(images, list):
                dropped_no_image += 1
                continue

            # 保持你训练逻辑简单：只吃单图样本（多图跳过）
            if len(images) != 1:
                dropped_multi_image += 1
                continue

            texts = ex.get("texts", None)
            if not texts or not isinstance(texts, list):
                dropped_bad_texts += 1
                continue

            # texts: list[{"user": "...", "assistant": "..."}, ...]
            messages: List[Dict[str, Any]] = []
            has_user = False
            has_asst = False

            for t in texts:
                if not isinstance(t, dict):
                    continue

                u = t.get("user", None)
                a = t.get("assistant", None)

                if isinstance(u, str) and u.strip():
                    has_user = True
                    messages.append({"role": "user", "content": _norm_img_token(u.strip())})

                if isinstance(a, str) and a.strip():
                    has_asst = True
                    messages.append({"role": "assistant", "content": _norm_img_token(a.strip())})

            if not (has_user and has_asst) or len(messages) < 2:
                dropped_bad_texts += 1
                continue

            # 如果 user 里完全没有图片占位，但样本确实有图片：强制在第一条 user 前补一个 <img>
            if IMG_TOKEN not in "".join(
                [m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
            ):
                for i, m in enumerate(messages):
                    if m.get("role") == "user":
                        m["content"] = (IMG_TOKEN + "\n" + m["content"]).strip()
                        break

            try:
                txt, tok_len = apply_template(tok, messages)
            except Exception:
                dropped_bad_texts += 1
                continue

            if tok_len > MAX_TOTAL_TOKENS:
                dropped_too_long += 1
                continue

            ret = save_image_2x2_plus_thumb(images[0], image_dir, cur_index)
            if ret is None:
                dropped_save_fail += 1
                continue

            img_files, vision_meta = ret
            rec = {
                "index": cur_index,
                "TXT": txt,
                "dataset": f"HuggingFaceM4/FineVision/{subset_name}",
                "source_index": orig_idx,
                "image_file": img_files,     # list[str]，5 张
                "length": tok_len,
                "vision_meta": vision_meta,  # 你原来的结构
                "lang": "en",
            }

            line = json.dumps(rec, ensure_ascii=False) + "\n"
            fout.write(line)
            total_bytes += len(line.encode("utf-8"))

            kept += 1
            cur_index += 1
            pbar.update(1)

        pbar.close()
        if kept >= max_samples:
            break

    print(
        f"  [FineVision/{subset_name}] kept={kept}, "
        f"dropped_no_image={dropped_no_image}, "
        f"dropped_multi_image={dropped_multi_image}, "
        f"dropped_bad_texts={dropped_bad_texts}, "
        f"dropped_too_long={dropped_too_long}, "
        f"dropped_save_fail={dropped_save_fail}, "
        f"dropped_decode_error={dropped_decode_error}"
    )
    return cur_index, kept, total_bytes


def process_scienceqa_train(
    tok: AutoTokenizer,
    fout,
    image_dir: Path,
    max_samples: int,
    start_index: int,
) -> Tuple[int, int, int]:
    print(">>> 加载 derek-thomas/ScienceQA (train)...")
    ds = load_dataset("derek-thomas/ScienceQA", split="train")
    try:
        ds = ds.cast_column("image", HFImage(decode=False))
    except Exception:
        pass
    indices = list(range(len(ds)))
    random.shuffle(indices)

    kept = 0
    total_bytes = 0
    cur_index = start_index

    dropped_no_image = 0
    dropped_no_solution = 0
    dropped_bad_choice = 0
    dropped_thought_too_long = 0
    dropped_total_too_long = 0
    dropped_save_fail = 0
    dropped_decode_error = 0
    for orig_idx in tqdm(indices, desc="ScienceQA/train"):
        if kept >= max_samples:
            break

        try:
            ex = ds[orig_idx]
        except Exception:
            dropped_decode_error += 1
            continue

        img_field = ex.get("image", None)
        if img_field is None:
            dropped_no_image += 1
            continue

        solution = (ex.get("solution") or "").strip()
        if not solution:
            dropped_no_solution += 1
            continue

        question = (ex.get("question") or "").strip()
        choices = ex.get("choices") or []
        answer_idx = ex.get("answer", None)

        if question == "" or not choices or answer_idx is None:
            dropped_bad_choice += 1
            continue

        if not (0 <= answer_idx < len(choices)):
            dropped_bad_choice += 1
            continue

        # 构造 user 文本：图片 + 题目 + 选项（可附加 hint）
        hint = (ex.get("hint") or "").strip()
        # lecture 一般很长、偏 background，这里就先不加了，让 COT 自己去补充推理
        parts: List[str] = [IMG_TOKEN]

        if hint:
            parts.append(f"Hint: {hint}")

        parts.append(f"Question: {question}")
        parts.append("Choices:")

        used_letters: List[str] = []
        for i, choice in enumerate(choices):
            if i >= len(SCIENCEQA_LETTERS):
                break
            letter_i = SCIENCEQA_LETTERS[i]
            used_letters.append(letter_i)
            parts.append(f"({letter_i}) {choice}")

        # 在 user 提示里明确要求用 boxed 形式输出
        if used_letters:
            parts.append(
                f"Put your final answer in LaTeX boxed form like $\\boxed{{answer}}$. "
            )

        user_text = "\n".join(parts).strip()

        # thought = solution（简短解释），final answer = 单独一行答案字母
        thought = solution
        letter = SCIENCEQA_LETTERS[answer_idx]
        answer = f"The answer is $\\boxed{{{letter}}}$."

        try:
            thought_len = count_thought_tokens(tok, thought)
        except Exception:
            dropped_thought_too_long += 1
            continue

        if thought_len > MAX_THOUGHT_TOKENS:
            dropped_thought_too_long += 1
            continue

        messages = build_messages_with_cot(
            user_text=user_text,
            answer=answer,
            thought=thought,
            use_system=True,
        )
        if messages is None:
            dropped_bad_choice += 1
            continue

        try:
            txt, total_len = apply_template(tok, messages)
        except Exception:
            dropped_total_too_long += 1
            continue

        if total_len > MAX_TOTAL_TOKENS:
            dropped_total_too_long += 1
            continue

        ret = save_image_2x2_plus_thumb(img_field, image_dir, cur_index)
        if ret is None:
            dropped_save_fail += 1
            continue

        img_files, vision_meta = ret

        rec = {
            "index": cur_index,
            "TXT": txt,
            "dataset": "derek-thomas/ScienceQA/train",
            "source_index": orig_idx,
            "image_file": img_files,  # <- list[str]
            "length": total_len,
            "lang": "en",
            "has_cot": True,
            "vision_meta": vision_meta,
            "cot_source": "scienceqa_solution",
        }

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        fout.write(line)
        total_bytes += len(line.encode("utf-8"))

        kept += 1
        cur_index += 1

    print(
        f"  [ScienceQA/train] kept={kept}, "
        f"dropped_no_image={dropped_no_image}, "
        f"dropped_no_solution={dropped_no_solution}, "
        f"dropped_bad_choice={dropped_bad_choice}, "
        f"dropped_thought_too_long={dropped_thought_too_long}, "
        f"dropped_total_too_long={dropped_total_too_long}, "
        f"dropped_save_fail={dropped_save_fail}",
        f"dropped_decode_error={dropped_decode_error}"
    )
    return cur_index, kept, total_bytes
# =====================================================
# 4) HuggingFaceH4/llava-instruct-mix-vsft 英文多轮视觉对话
#    字段: messages(list[{role, content[list]}]), images(list[Image])
# =====================================================

def convert_vsft_messages(raw_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], bool]:
    """
    把 vSFT 的 message 格式转成 OAI 式 messages：
    - 把所有 image 块变成一个统一的 <img> 占位
    - 其他 text 块直接拼接
    返回: (处理后的 messages, 是否出现过图片)
    """
    out_messages: List[Dict[str, str]] = []
    has_image_any = False

    for m in raw_messages:
        role_raw = (m.get("role") or "").lower()
        content_list = m.get("content") or []

        text_parts: List[str] = []
        has_image_this = False

        for chunk in content_list:
            t = chunk.get("type", None)
            if t == "text":
                txt = chunk.get("text", None)
                if txt:
                    text_parts.append(txt)
            elif t == "image":
                text_parts.append(IMG_TOKEN)
                has_image_this = True
                has_image_any = True

        merged = "".join(text_parts).strip()
        if not merged:
            continue

        role = "assistant" if role_raw == "assistant" else "user"
        out_messages.append({"role": role, "content": merged})

    return out_messages, has_image_any
def process_vsft(
    tok: AutoTokenizer,
    fout,
    image_dir: Path,
    max_samples: int,
    start_index: int,
) -> Tuple[int, int, int]:
    """
    处理 HuggingFaceH4/llava-instruct-mix-vsft：
    - 把 messages 转成 OAI 格式
    - 只保留带图片的样本，且强制单图（len(images) == 1）
    - 文本通过你的 chat_template 渲染到 TXT
    """
    print(">>> 加载 HuggingFaceH4/llava-instruct-mix-vsft (train)...")
    ds = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train")

    indices = list(range(len(ds)))
    random.shuffle(indices)

    kept = 0
    total_bytes = 0
    cur_index = start_index

    dropped_no_image = 0
    dropped_multi_image = 0
    dropped_bad_msg = 0
    dropped_too_long = 0
    dropped_save_fail = 0

    for orig_idx in tqdm(indices, desc="vSFT/en"):
        if kept >= max_samples:
            break

        ex = ds[orig_idx]

        images_field = ex.get("images", None)
        if not images_field or len(images_field) == 0:
            dropped_no_image += 1
            continue

        # 多图样本先全部跳过，简化训练逻辑
        if len(images_field) != 1:
            dropped_multi_image += 1
            continue

        raw_messages = ex.get("messages", None)
        if not raw_messages:
            dropped_bad_msg += 1
            continue

        messages, has_image_any = convert_vsft_messages(raw_messages)
        if not messages or not has_image_any:
            dropped_bad_msg += 1
            continue

        try:
            txt, tok_len = apply_template(tok, messages)
        except Exception:
            dropped_bad_msg += 1
            continue

        if tok_len > MAX_TOTAL_TOKENS:
            dropped_too_long += 1
            continue

        first_img = images_field[0]
        ret = save_image_2x2_plus_thumb(first_img, image_dir, cur_index)
        if ret is None:
            dropped_save_fail += 1
            continue

        img_files, vision_meta = ret

        if img_files is None:
            dropped_save_fail += 1
            continue

        rec = {
            "index": cur_index,
            "TXT": txt,
            "dataset": "HuggingFaceH4/llava-instruct-mix-vsft",
            "source_index": orig_idx,
            "image_file": img_files,
            "length": tok_len,
            "vision_meta": vision_meta,
            "lang": "en",
        }

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        fout.write(line)
        total_bytes += len(line.encode("utf-8"))

        kept += 1
        cur_index += 1

    print(
        f"  [vSFT] kept={kept}, "
        f"dropped_no_image={dropped_no_image}, "
        f"dropped_multi_image={dropped_multi_image}, "
        f"dropped_bad_msg={dropped_bad_msg}, "
        f"dropped_too_long={dropped_too_long}, "
        f"dropped_save_fail={dropped_save_fail}"
    )
    return cur_index, kept, total_bytes
# =====================================================
# 5) yuecao0119/MMInstruct-GPT4V caption_cn：中文图像描述/对话
#    字段: image(Image), conversations(list[{from,value}])
# =====================================================

def process_caption_cn(
    tok: AutoTokenizer,
    fout,
    image_dir: Path,
    max_samples: int,
    start_index: int,
) -> Tuple[int, int, int]:
    print(">>> 加载 yuecao0119/MMInstruct-GPT4V (caption_cn, train)...")
    ds = load_dataset("yuecao0119/MMInstruct-GPT4V", "caption_cn", split="train")
    try:
        ds = ds.cast_column("image", HFImage(decode=False))
    except Exception:
        pass

    indices = list(range(len(ds)))
    random.shuffle(indices)

    kept = 0
    total_bytes = 0
    cur_index = start_index

    dropped_no_image = 0
    dropped_bad_conv = 0
    dropped_too_long = 0
    dropped_save_fail = 0
    dropped_decode_error = 0

    for orig_idx in tqdm(indices, desc="MMInstruct/caption_cn"):
        if kept >= max_samples:
            break

        try:
            ex = ds[orig_idx]
        except Exception:
            dropped_decode_error += 1
            continue

        img_field = ex.get("image", None)
        if img_field is None:
            dropped_no_image += 1
            continue

        conv = ex.get("conversations", None)
        if not conv or len(conv) < 2:
            dropped_bad_conv += 1
            continue

        human_msg = next((c for c in conv if c.get("from") == "human"), None)
        gpt_msg   = next((c for c in conv if c.get("from") == "gpt"), None)
        if human_msg is None or gpt_msg is None:
            dropped_bad_conv += 1
            continue

        user_text_raw = human_msg.get("value", "") or ""
        assistant_text = gpt_msg.get("value", "") or ""

        if not user_text_raw.strip() or not assistant_text.strip():
            dropped_bad_conv += 1
            continue

        user_text = user_text_raw.replace("<image>", IMG_TOKEN)

        messages = [
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]

        try:
            txt, tok_len = apply_template(tok, messages)
        except Exception:
            dropped_bad_conv += 1
            continue

        if tok_len > MAX_TOTAL_TOKENS:
            dropped_too_long += 1
            continue

        ret = save_image_2x2_plus_thumb(img_field, image_dir, cur_index)
        if ret is None:
            dropped_save_fail += 1
            continue

        img_files, vision_meta = ret
        rec = {
            "index": cur_index,
            "TXT": txt,
            "dataset": "yuecao0119/MMInstruct-GPT4V/caption_cn",
            "source_index": orig_idx,
            "image_file": img_files,  # <- list[str]
            "length": tok_len,
            "vision_meta": vision_meta,
            "lang": "zh",
        }

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        fout.write(line)
        total_bytes += len(line.encode("utf-8"))

        kept += 1
        cur_index += 1

    print(
        f"  [MMInstruct caption_cn] kept={kept}, "
        f"dropped_no_image={dropped_no_image}, "
        f"dropped_bad_conv={dropped_bad_conv}, "
        f"dropped_too_long={dropped_too_long}, "
        f"dropped_save_fail={dropped_save_fail}, "
        f"dropped_decode_error={dropped_decode_error}"
    )
    return cur_index, kept, total_bytes


# =====================================================
# 主流程
# =====================================================
def main():
    random.seed(RNG_SEED)

    print(">>> 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.chat_template is None:
        raise ValueError(
            "当前 tokenizer.chat_template 为空。\n"
            "请先把带 <|thought_start|>/<|thought_end|> 支持的 Jinja 模板写进 tokenizer 再跑。"
        )

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_ROOT.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 规划 part 顺序（保持你原来的逻辑顺序）
    # ----------------------------
    part_plan: list[tuple[str, Any]] = []

    # # 2) Xkev/LLaVA-CoT-100k
    # def _run_xkev(tok, fout, image_dir, start_index):
    #     if USE_FULL_XKEV:
    #         max_xkev = 60_000_000
    #     else:
    #         max_xkev = NUM_XKEV
    #     return process_xkev_llava_cot(tok, fout, image_dir, max_xkev, start_index)
    #
    # part_plan.append(("xkev_llava_cot_100k", _run_xkev))
    #
    # # 1) OneVision: 每个子集一个 part
    # for subset_name, max_samples in ONEVISION_SUBSETS.items():
    #     if subset_name == "scienceqa":
    #         repeat = 10
    #         max_samples_effective = 10_000_000
    #     else:
    #         repeat = 1
    #         max_samples_effective = max_samples
    #
    #     part_name = f"onevision_{subset_name}"
    #
    #     def _make_runner(_subset=subset_name, _max=max_samples_effective, _repeat=repeat):
    #         def _runner(tok, fout, image_dir, start_index):
    #             return process_onevision_subset(
    #                 tok,
    #                 fout,
    #                 image_dir,
    #                 _subset,
    #                 _max,
    #                 start_index,
    #                 repeat=_repeat,
    #             )
    #         return _runner
    #
    #     part_plan.append((part_name, _make_runner()))
    #
    # # 3) ScienceQA train
    # def _run_sqa(tok, fout, image_dir, start_index):
    #     return process_scienceqa_train(tok, fout, image_dir, NUM_SCIENCEQA_TRAIN, start_index)
    #
    # part_plan.append(("scienceqa_train", _run_sqa))
    #
    # # 4) vSFT
    # def _run_vsft(tok, fout, image_dir, start_index):
    #     return process_vsft(tok, fout, image_dir, NUM_VSFT_EN, start_index)
    #
    # part_plan.append(("vsft_en", _run_vsft))
    #
    # # 5) caption_cn
    # def _run_cn(tok, fout, image_dir, start_index):
    #     return process_caption_cn(tok, fout, image_dir, NUM_CAPTION_CN, start_index)
    #
    # part_plan.append(("mminstruct_caption_cn", _run_cn))


    def _make_fv_runner(_subset: str, _max: int):
        def _runner(tok, fout, image_dir, start_index):
            return process_finevision_subset(
                tok=tok,
                fout=fout,
                image_dir=image_dir,
                subset_name=_subset,
                max_samples=_max,
                start_index=start_index,
                repeat=1,
            )

        return _runner

    # 为了可控比例：按三个 dict 的顺序 append
    for subset_name, max_samples in FINEVISION_OCR_SUBSETS.items():
        part_plan.append((f"finevision_{subset_name}", _make_fv_runner(subset_name, max_samples)))

    for subset_name, max_samples in FINEVISION_GENERAL_SUBSETS.items():
        part_plan.append((f"finevision_{subset_name}", _make_fv_runner(subset_name, max_samples)))

    for subset_name, max_samples in FINEVISION_SCIENCE_FRIENDLY_SUBSETS.items():
        part_plan.append((f"finevision_{subset_name}", _make_fv_runner(subset_name, max_samples)))
    # ----------------------------
    # 安全检查：不允许“后面的 part done，但前面的 part 没 done”
    # 否则你重跑前面的 part 会覆盖后面 part 的全局 index 图片
    # ----------------------------
    done_flags: list[bool] = []
    part_names: list[str] = []
    for pn, _ in part_plan:
        part_names.append(pn)
        _, jsonl_final, _, success_txt = _part_paths(pn)
        ok = _read_success(success_txt)
        done_flags.append(bool(ok and ok.get("ok") is True and jsonl_final.exists()))

    seen_not_done = False
    for pn, is_done in zip(part_names, done_flags):
        if not is_done:
            seen_not_done = True
        elif seen_not_done:
            raise RuntimeError(
                f"检测到顺序不一致：'{pn}' 已完成，但它前面有未完成 part。\n"
                f"为避免覆盖全局 images index，请先把后续已完成 part 的目录删掉（vlm_sft_parts 下对应子目录），再重跑。"
            )

    # ----------------------------
    # 逐个 part 跑：全局 index 连续增长（图片直接写到 IMAGE_DIR）
    # ----------------------------
    global_index = 0
    total_kept = 0
    total_bytes = 0

    for part_name, runner in part_plan:
        next_index, kept, bts, skipped = run_part(
            part_name=part_name,
            runner_fn=runner,
            tok=tok,
            image_dir=IMAGE_DIR,      # ✅ 关键：所有 part 都写进同一个 IMAGE_DIR
            start_index=global_index, # ✅ 关键：全局 index
        )
        global_index = next_index
        total_kept += kept
        total_bytes += bts

    # ----------------------------
    # 合并：只 concat jsonl（不用动 images）
    # ----------------------------
    merge_parts_jsonl(part_names, out_path, IMAGE_DIR)
    shuf_path = out_path.with_name(out_path.stem + ".shuf" + out_path.suffix)
    shuffle_jsonl_buckets(out_path, shuf_path, seed=RNG_SEED, num_buckets=256)
    mb_all = total_bytes / (1024 * 1024)
    print(">>> 完成 VLM SFT 数据打包（分块 + 全局 images + 合并 JSONL）！")
    print(f"    写入样本总数(统计口径=各part kept之和): {total_kept}")
    print(f"    最终 global_index = {global_index} (图片名从 0.jpg 到 {global_index-1}.jpg)")
    print(f"    估算 JSONL 体积(各part累计): {mb_all:.2f} MB")
    print(f"    输出 JSONL: {out_path.resolve()}")
    print(f"    图片目录: {IMAGE_DIR.resolve()}")

if __name__ == "__main__":
    main()
