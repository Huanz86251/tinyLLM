"""Generate the MiniCPM teacher-logit LMDB used by pretraining KD."""

from _run_module import run


if __name__ == "__main__":
    run("train/pretrain/build_teacher_logits.py")

