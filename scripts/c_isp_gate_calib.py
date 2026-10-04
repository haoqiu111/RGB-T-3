# -*- coding: utf-8 -*-
"""Calibrate the gate threshold: on the LYNRED test split compute the distribution of the monotone-fit residual
for clean frames (camera 8-bit output), the five real AGC kinds and the infrared/registration synthetic degradations.
Usage: python c_isp_gate_calib.py --limit 150   Output: runs/c_invert/gate_calib.json"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402
from cdkit.ispgate import monotone_residual  # noqa: E402
from cdkit.realdeg import REAL_KINDS, apply_any  # noqa: E402
from cdkit.thermal16 import tone_map  # noqa: E402
from cdkit.ispgate import _resize  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from rgbta.data.degradations import ALL_KINDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=150); ap.add_argument("--domain", default="lynred_mds")
    a = ap.parse_args()
    ids = (SPLITS / f"{a.domain}_test.txt").read_text(encoding="utf-8").split()[: a.limit]
    ds = UnifiedPairedDataset(a.domain, ids=ids, label_mode="super", keep_empty=False); raw = RawAccess(a.domain)
    res = defaultdict(list); rng = np.random.default_rng(0)
    for i in range(len(ds)):
        vis, ir, t = ds[i]; v = vis.permute(1, 2, 0).numpy(); g = ir[0].numpy(); r16 = raw(t["image_id"])
        if r16 is None:
            continue
        res["clean_shipped"].append(monotone_residual(g, r16))
        canon = _resize(tone_map(r16, "linear_pct:1"), (g.shape[1], g.shape[0]))
        res["clean_canonical"].append(monotone_residual(canon, r16))
        for k in REAL_KINDS:
            _, d = apply_any(k, 0, v, g, rng, r16); res[k].append(monotone_residual(d, r16))
        for k in ALL_KINDS:
            if not (k.startswith("ir") or k == "misalignment"):
                continue
            for sev in (2, 4):
                _, d = apply_any(k, sev, v, g, rng, r16); res[f"{k}_s{sev}"].append(monotone_residual(d, r16))
    out = Path(RUNS) / "c_invert"; out.mkdir(exist_ok=True)
    summ = {k: {"n": len(v), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "min": float(np.min(v))} for k, v in res.items()}
    (out / "gate_calib.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    for k, s in summ.items():
        print(f"{k:24s} n={s['n']:4d} p50={s['p50']:.4f} p95={s['p95']:.4f} p99={s['p99']:.4f} min={s['min']:.4f}")


if __name__ == "__main__":
    main()
