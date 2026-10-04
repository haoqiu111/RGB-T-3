# -*- coding: utf-8 -*-
"""Hold-out evaluators #2/#3 for a new domain: RT-DETR-L and YOLO11m (ultralytics, fine-tuned from COCO weights).
The base checkpoints rtdetr-l.pt / yolo11m.pt are expected under SFA_WEIGHTS.
Usage: python c_train_ultra.py --domain lynred_mds --arch rtdetr|yolo --epochs 20
Output: runs/heldout/<domain>_<arch>/weights/best.pt"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cdkit.paths import RUNS, WEIGHTS  # noqa: E402

RUNS = Path(RUNS)
A_SCRIPTS = Path(WEIGHTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--arch", choices=["rtdetr", "yolo"], required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume from runs/heldout/<domain>_<arch>/weights/last.pt")
    args = ap.parse_args()
    last = RUNS / "heldout" / f"{args.domain}_{args.arch}" / "weights" / "last.pt"
    if args.arch == "rtdetr":
        from ultralytics import RTDETR as M
        batch = args.batch or 8; base = A_SCRIPTS / "rtdetr-l.pt"
    else:
        from ultralytics import YOLO as M
        batch = args.batch or 16; base = A_SCRIPTS / "yolo11m.pt"
    if args.resume and last.exists():
        M(str(last)).train(resume=True)
        return
    M(str(base)).train(data=str(RUNS / "heldout" / f"{args.domain}_yolo_data" / "data.yaml"), epochs=args.epochs, batch=batch,
                       imgsz=640, device=0, project=str(RUNS / "heldout"), name=f"{args.domain}_{args.arch}", exist_ok=True,
                       workers=6, patience=8, seed=1, verbose=True)


if __name__ == "__main__":
    main()
