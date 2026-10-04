# -*- coding: utf-8 -*-
"""Select the margin threshold tau of router v2 on the validation split (ported from select_tau_v2.py; test is never used).
Usage: python c_select_tau.py --domain lynred_mds --seed 0
Output: runs/router/<domain>_router_v2_s<seed>/tau.json"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import RUNS  # noqa: E402
from rgbta.models.router import RouterNetV2  # noqa: E402
from rgbta.utils.common import get_logger  # noqa: E402
from c_train_router import DomainRouterDataset  # noqa: E402
from train_router import collate  # noqa: E402

TAUS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--variant", default="")
    args = ap.parse_args(); device = "cuda"
    out = Path(RUNS) / "router" / f"{args.domain}_router_v2{args.variant}_s{args.seed}"
    log = get_logger(f"select_tau_{args.domain}_s{args.seed}", str(out / "select_tau.log"))
    model = RouterNetV2().to(device).eval()
    model.load_state_dict(torch.load(out / "best.pth", map_location="cpu", weights_only=False)["model"])
    va = DomainRouterDataset(args.domain, "val", args.variant)
    lva = DataLoader(va, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)
    recs, ptr = [], 0
    with torch.no_grad():
        for vg, ir, metas in lva:
            vg, ir = vg.to(device), ir.to(device); qhat = model(vg, ir)
            H8, W8 = qhat.shape[-2:]; sy, sx = H8 / vg.shape[-2], W8 / vg.shape[-1]
            for bi_, (boxes, _a, _bq, qv) in enumerate(metas):
                stress = va.samples[va.keys[ptr]]["kind"] != "clean"; ptr += 1
                for k in range(len(boxes)):
                    x1 = int(boxes[k, 0] * sx); y1 = int(boxes[k, 1] * sy)
                    x2 = max(int(boxes[k, 2] * sx), x1 + 1); y2 = max(int(boxes[k, 3] * sy), y1 + 1)
                    recs.append((qhat[bi_, :, y1:y2, x1:x2].mean(dim=(1, 2)).cpu().numpy(), qv[k].numpy(), stress))
    pred = np.array([r[0] for r in recs]); qv = np.array([r[1] for r in recs]); stress = np.array([r[2] for r in recs])
    log.info(f"val boxes={len(recs)} stress_frac={stress.mean():.2f}")
    rows, best = [], None
    for tau in TAUS:
        alt = pred[:, 1:].max(axis=1); a_alt = 1 + pred[:, 1:].argmax(axis=1)
        act = np.where(alt - pred[:, 0] >= tau, a_alt, 0)
        gain = qv[np.arange(len(qv)), act] - qv[:, 0]
        row = {"tau": tau, "clean_gain": float(gain[~stress].mean()), "stress_gain": float(gain[stress].mean()),
               "switch_rate": float((act > 0).mean())}
        rows.append(row); log.info(json.dumps(row))
        if row["clean_gain"] >= -0.005 and (best is None or row["stress_gain"] > best["stress_gain"]):
            best = row
    if best is None:
        best = max(rows, key=lambda r: r["clean_gain"])
    (out / "tau.json").write_text(json.dumps({"tau": best["tau"], "sweep": rows, "selected": best}, indent=2), encoding="utf-8")
    log.info(f"selected tau={best['tau']} ({best})")


if __name__ == "__main__":
    main()
