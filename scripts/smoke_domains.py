# -*- coding: utf-8 -*-
"""Smoke test of the domain loaders: samples images, checks the tensor protocol and box validity, and writes vis|ir overlays for a visual registration check.

Usage: python smoke_domains.py --domains lynred_mds,smod  --n 3
Output: runs/smoke_domains/<domain>_<k>.jpg ; summary on stdout
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401,E402
from cdkit import INDEX, RUNS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402


def draw(vis, ir, boxes, labels, classes, path):
    v = Image.fromarray((vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    r = Image.fromarray((ir[0].numpy() * 255).astype(np.uint8)).convert("RGB")
    for im in (v, r):
        d = ImageDraw.Draw(im)
        for b, l in zip(boxes.numpy(), labels.numpy()):
            d.rectangle(b.tolist(), outline=(255, 0, 0), width=2)
            d.text((b[0] + 2, b[1] + 1), classes[int(l)] if classes else str(int(l)), fill=(255, 255, 0))
    out = Image.new("RGB", (v.width * 2, v.height)); out.paste(v, (0, 0)); out.paste(r, (v.width, 0))
    out.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ir-source", default="8bit")
    args = ap.parse_args()
    doms = [d for d in args.domains.split(",") if d] or sorted(p.stem for p in INDEX.glob("*.json"))
    out = Path(RUNS) / "smoke_domains"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for d in doms:
        try:
            ds = UnifiedPairedDataset(d, ir_source=args.ir_source)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {d}: {e!r}"); continue
        labeled = [i for i, r in enumerate(ds.recs) if len(r["boxes"])]
        pool = labeled if labeled else list(range(len(ds)))
        pick = rng.choice(pool, size=min(args.n, len(pool)), replace=False)
        bad = 0
        for k, i in enumerate(pick):
            vis, ir, t = ds[int(i)]
            assert vis.shape == (3, 512, 640) and ir.shape == (1, 512, 640), (d, vis.shape, ir.shape)
            b = t["boxes"]
            if len(b):
                bad += int(((b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])).sum())
            draw(vis, ir, t["boxes"], t["labels"], ds.classes, out / f"{d}_{k}.jpg")
        print(f"[ok] {d}: n={len(ds)} labeled={len(labeled)} sampled={len(pick)} bad_boxes={bad} "
              f"ir_range=[{float(ir.min()):.2f},{float(ir.max()):.2f}] vis_mean={float(vis.mean()):.2f}")


if __name__ == "__main__":
    main()
