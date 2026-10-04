# -*- coding: utf-8 -*-
"""Build YOLO-format data for the ultralytics RT-DETR / YOLO evaluators (new domain, super classes person/car/bicycle).
Every frozen-split id is written twice (vis and ir, 640x512, same labels) = modality-agnostic evaluator; clean frames only.
Usage: python c_prep_yolo_data.py --domain lynred_mds
Output: runs/heldout/<domain>_yolo_data/{images,labels}/{train,val} + data.yaml"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", required=True); args = ap.parse_args()
    out = Path(RUNS) / "heldout" / f"{args.domain}_yolo_data"
    for tag in ("train", "val"):
        ids = (SPLITS / f"{args.domain}_{tag}.txt").read_text(encoding="utf-8").split()
        ds = UnifiedPairedDataset(args.domain, ids=ids, label_mode="super", keep_empty=False)
        img_dir, lbl_dir = out / "images" / tag, out / "labels" / tag
        img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for i in range(len(ds)):
            vis, ir, t = ds[i]
            W, H = vis.shape[-1], vis.shape[-2]
            lines = []
            for b, l in zip(t["boxes"].numpy(), t["labels"].numpy()):
                cx, cy, bw, bh = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H
                lines.append(f"{int(l)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            for mod, arr in (("vis", vis.permute(1, 2, 0).numpy()), ("ir", ir[0].numpy()[..., None].repeat(3, axis=2))):
                p = img_dir / f"{t['image_id']}__{mod}.jpg"
                if not p.exists():
                    Image.fromarray((arr * 255).astype(np.uint8)).save(p, quality=95)
                (lbl_dir / f"{t['image_id']}__{mod}.txt").write_text("\n".join(lines), encoding="utf-8")
                n += 1
        print(f"{tag}: {n} images")
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: car\n  2: bicycle\n", encoding="utf-8")
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
