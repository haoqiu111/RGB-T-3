# -*- coding: utf-8 -*-
"""Counterfactual evidence audit on LYNRED-MDS with the in-domain hold-out RetinaNet evaluator.
Intervention A: erase the infrared evidence inside the box (median fill); intervention B: destroy the visible texture
inside the box (strong blur + contrast compression).
For composed (decision-level stitched image) / fused / ir / vis report delta q, drop rate and true-hallucination rate.
Usage: python c_counterfactual.py --limit 400 --select thermal_dep|largest   Output: runs/counterfactual_c/<select>/report.json"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from rgbta.models.router import compose  # noqa: E402
from rgbta.utils.common import get_logger  # noqa: E402
from c_eval import DomRetina, load_system  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402


def erase_ir_box(ir, box):
    out = ir.copy(); x1, y1, x2, y2 = [int(v) for v in box]; med = float(np.median(ir)); rng = np.random.default_rng(0)
    out[y1:y2, x1:x2] = np.clip(med + rng.normal(0, 0.01, out[y1:y2, x1:x2].shape), 0, 1).astype(np.float32); return out


def degrade_vis_box(vis, box):
    out = vis.copy(); x1, y1, x2, y2 = [int(v) for v in box]; patch = out[y1:y2, x1:x2]
    if patch.size == 0:
        return out
    for c in range(3):
        patch[..., c] = gaussian_filter(patch[..., c], 4.0)
    m = patch.mean(); out[y1:y2, x1:x2] = np.clip((patch - m) * 0.3 + m, 0, 1); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", default="lynred_mds"); ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--select", choices=["largest", "thermal_dep"], default="thermal_dep"); ap.add_argument("--tau", type=float, default=0.04)
    args = ap.parse_args(); device = "cuda"
    out = Path(RUNS) / "counterfactual_c" / args.select; out.mkdir(parents=True, exist_ok=True); log = get_logger("cf_c", str(out / "run.log"))
    critics, fusion, router, _ = load_system(args.domain, 0, device); tau = args.tau; ev = DomRetina(args.domain, device)

    @torch.no_grad()
    def make_outputs(v_np, i_np):
        vt = torch.from_numpy(np.ascontiguousarray(v_np.transpose(2, 0, 1)))[None].to(device); it_ = torch.from_numpy(np.ascontiguousarray(i_np[None][None])).to(device)
        vg = rgb_to_gray(vt); hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
        fused = fusion(vg, it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw))
        qhat = router(vg, it_); alt = qhat[:, 1:].max(dim=1, keepdim=True).values
        cell = torch.where((alt - qhat[:, :1]) >= tau, 1 + qhat[:, 1:].argmax(dim=1, keepdim=True), torch.zeros_like(alt, dtype=torch.long))
        composed, _ = compose(torch.zeros_like(qhat).scatter_(1, cell, 10.0), fused, vg, it_, hard=True)
        return {"composed": composed.clamp(0, 1).repeat(1, 3, 1, 1), "fused": fused.clamp(0, 1).repeat(1, 3, 1, 1), "ir": it_.repeat(1, 3, 1, 1), "vis": vt}

    ids = (SPLITS / f"{args.domain}_test.txt").read_text(encoding="utf-8").split()
    ds = UnifiedPairedDataset(args.domain, ids=ids, label_mode="super", keep_empty=False)
    loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_pairs); rows = []
    with torch.no_grad():
        for bi, (vis, ir, targets) in enumerate(loader):
            if bi >= args.limit:
                break
            t = targets[0]
            if len(t["boxes"]) == 0:
                continue
            v_np = vis[0].permute(1, 2, 0).numpy(); i_np = ir[0, 0].numpy(); base = make_outputs(v_np, i_np)
            q0 = {c: ev.box_q(img, t) for c, img in base.items()}
            if any(v is None for v in q0.values()):
                continue
            if args.select == "thermal_dep":
                cand = [j for j in range(len(t["boxes"])) if q0["ir"][j] > 0.2]
                if not cand:
                    continue
                k = min(cand, key=lambda j: q0["vis"][j])
            else:
                k = int(np.argmax((t["boxes"][:, 2] - t["boxes"][:, 0]) * (t["boxes"][:, 3] - t["boxes"][:, 1])))
            box = t["boxes"][k].tolist()
            for interv, (v2, i2) in {"erase_ir": (v_np, erase_ir_box(i_np, box)), "degrade_vis": (degrade_vis_box(v_np, box), i_np)}.items():
                pert = make_outputs(v2, i2); q1 = {c: ev.box_q(img, t) for c, img in pert.items()}
                if any(v is None for v in q1.values()):
                    continue
                for c in ("composed", "fused", "ir", "vis"):
                    rows.append({"image_id": t["image_id"], "intervention": interv, "channel": c, "q_before": round(float(q0[c][k]), 4),
                                 "q_after": round(float(q1[c][k]), 4), "delta": round(float(q1[c][k] - q0[c][k]), 4), "q_vis_before": round(float(q0["vis"][k]), 4)})
            if (bi + 1) % 50 == 0:
                log.info(f"{bi+1}/{min(args.limit, len(loader))}")
    with open(out / "per_box.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    report = {}
    for interv in ("erase_ir", "degrade_vis"):
        for c in ("composed", "fused", "ir", "vis"):
            sub = [r for r in rows if r["intervention"] == interv and r["channel"] == c and r["q_before"] > 0.2]
            if not sub:
                continue
            d = np.array([r["delta"] for r in sub]); e = {"n": len(sub), "delta_mean": float(d.mean()), "drop_rate": float((d < -0.02).mean()), "big_drop_rate": float((d < -0.2).mean())}
            if interv == "erase_ir":
                blind = [r for r in sub if r["q_vis_before"] < 0.2]
                if blind:
                    db = np.array([r["delta"] for r in blind]); e["n_vis_blind"] = len(blind); e["true_hallucination_rate"] = float((db > -0.02).mean())
                e["persist_rate_all"] = float((d > -0.02).mean())
            report[f"{interv}/{c}"] = e
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); log.info(json.dumps(report, indent=2)); log.info("done")


if __name__ == "__main__":
    main()
