# -*- coding: utf-8 -*-
"""Per-frame distribution of the two statistics seen by the blind two-level gate on the final-evaluation sampling
(for the gate-distribution figure and the trigger-rate table).
For every LYNRED-MDS test image replay c_eval's clean + 3 stress draws (same rng key) and record
  resid = monotone-fit residual (8-bit against raw counts), dev = mean absolute deviation from the camera map (emulator).
CPU only. Usage: python c_gate_dist.py   Output: runs/c_invert/gate_dist.json"""
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402
from cdkit.ispgate import emulator_map, monotone_residual  # noqa: E402
from cdkit.realdeg import apply_any, menu as build_menu  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from precompute_region_labels import draw_rng  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=0, help=">0: first N images in id-hash order (spot check of the training split)")
    args = ap.parse_args()
    dom = "lynred_mds"; ids = (SPLITS / f"{dom}_{args.split}.txt").read_text(encoding="utf-8").split()
    if args.limit:
        import hashlib
        ids = sorted(ids, key=lambda i: hashlib.sha256(f"gatedist:{i}".encode()).hexdigest())[: args.limit]
    ds = UnifiedPairedDataset(dom, ids=ids, label_mode="super", keep_empty=False); raw = RawAccess(dom); m = build_menu(with_real=True)
    rows = []
    for i in range(len(ds)):
        vis, ir, t = ds[i]
        if len(t["boxes"]) == 0:
            continue
        v = vis.permute(1, 2, 0).numpy(); g = ir[0].numpy(); r16 = raw(t["image_id"])
        if r16 is None:
            continue
        H, W = g.shape; emu = emulator_map(r16, (W, H))
        for draw in range(4):
            rng = draw_rng(f"ceval:{dom}:{t['image_id']}", draw)
            if draw == 0:
                kind, sev, idg = "clean", 0, g
            else:
                kind, sev = m[int(rng.integers(0, len(m)))]
                _, idg = apply_any(kind, sev, v, g, rng, r16)
            rows.append({"image_id": t["image_id"], "kind": kind, "severity": sev, "resid": round(float(monotone_residual(idg, r16)), 5),
                         "dev": round(float(np.abs(idg - emu).mean()), 5)})
        if (i + 1) % 100 == 0:
            print(i + 1, "/", len(ds), flush=True)
    out = Path(RUNS) / "c_invert"; out.mkdir(exist_ok=True)
    (out / ("gate_dist.json" if args.split == "test" else f"gate_dist_{args.split}.json")).write_text(json.dumps(rows), encoding="utf-8")
    kinds = sorted({r["kind"] for r in rows})
    for k in kinds:
        a = np.array([[r["resid"], r["dev"]] for r in rows if r["kind"] == k])
        fire = float(((a[:, 0] <= 0.02) & (a[:, 1] >= 0.10)).mean())
        print(f"{k:22s} n={len(a):5d} resid p50={np.median(a[:, 0]):.4f} max={a[:, 0].max():.4f} min={a[:, 0].min():.4f} | dev p50={np.median(a[:, 1]):.3f} | pass1={float((a[:, 0] <= 0.02).mean()):.3f} fire={fire:.3f}")


if __name__ == "__main__":
    main()
