# -*- coding: utf-8 -*-
"""Conformal risk control (CRC) calibration of ACCEPT/ABSTAIN (ported from the companion project's
calibrate_abstain.py) on the new domain with the real + synthetic menu.
Protocols: shifted (calib from train scenes -> official test) and exchangeable (official test scene groups split in half);
unit box / group (scene group).
Usage: python c_calibrate.py --domain lynred_mds --seed 0 --protocol exchangeable --unit group --menu both
Output: runs/abstain/<domain>_s<seed>_<protocol>_<unit>_<menu>[_gf<N>]/{calibration.json,test_eval.json,rc_curve.csv}"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from cdkit.realdeg import apply_any, menu as build_menu  # noqa: E402
from rgbta.eval.conformal import (crc_threshold, crc_threshold_grouped, evaluate_threshold,  # noqa: E402
                                  evaluate_threshold_grouped, risk_coverage_sweep)
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from rgbta.models.router import compose  # noqa: E402
from rgbta.utils.common import get_logger  # noqa: E402
from c_eval import load_system  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from precompute_region_labels import cand_box_q, draw_rng  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402

ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.30)


def collect(domain, ds, draws, device, nets, m, raw, log, tag):
    router, fusion, critics, tau = nets
    loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_pairs)
    rows = []
    with torch.no_grad():
        for bi, (vis, ir, targets) in enumerate(loader):
            t = targets[0]
            if len(t["boxes"]) == 0:
                continue
            v_np = vis[0].permute(1, 2, 0).numpy(); i_np = ir[0, 0].numpy(); raw16 = raw(t["image_id"])
            for draw in range(draws + 1):
                rng = draw_rng(f"ccal:{domain}:{t['image_id']}", draw)
                if draw == 0:
                    kind, sev, vd, idg = "clean", 0, v_np, i_np
                else:
                    kind, sev = m[int(rng.integers(0, len(m)))]
                    if kind.startswith("real_agc") and raw16 is None:
                        kind, sev = m[0]
                    vd, idg = apply_any(kind, sev, v_np, i_np, rng, raw16)
                vt = torch.from_numpy(np.ascontiguousarray(vd.transpose(2, 0, 1)))[None].to(device)
                it_ = torch.from_numpy(np.ascontiguousarray(idg[None][None])).to(device)
                vg = rgb_to_gray(vt); hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
                fused = fusion(vg, it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw))
                qhat = router(vg, it_)
                alt = qhat[:, 1:].max(dim=1, keepdim=True).values
                cell = torch.where((alt - qhat[:, :1]) >= tau, 1 + qhat[:, 1:].argmax(dim=1, keepdim=True), torch.zeros_like(alt, dtype=torch.long))
                logits = torch.zeros_like(qhat).scatter_(1, cell, 10.0); rel = qhat.gather(1, cell)
                composed, _ = compose(logits, fused, vg, it_, hard=True)
                q_c = cand_box_q(critics, composed.clamp(0, 1).repeat(1, 3, 1, 1), t)
                if q_c is None:
                    continue
                H8, W8 = rel.shape[-2:]; sy, sx = H8 / vg.shape[-2], W8 / vg.shape[-1]
                for k in range(len(q_c)):
                    b = t["boxes"][k]; x1, y1 = int(b[0] * sx), int(b[1] * sy)
                    x2, y2 = max(int(b[2] * sx), x1 + 1), max(int(b[3] * sy), y1 + 1)
                    rows.append({"image_id": t["image_id"], "kind": kind, "severity": sev, "box_idx": k,
                                 "score": float(rel[0, 0, y1:y2, x1:x2].mean()), "loss": 1.0 - float(q_c[k])})
            if (bi + 1) % 200 == 0:
                log.info(f"{tag}: {bi+1}/{len(loader)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--system", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--protocol", choices=["shifted", "exchangeable"], default="exchangeable")
    ap.add_argument("--unit", choices=["box", "group"], default="group"); ap.add_argument("--menu", choices=["synthetic", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N images of each set")
    ap.add_argument("--group-frames", type=int, default=0, help=">0: fine-grained groups of N consecutive frames within a sequence")
    args = ap.parse_args(); device = "cuda"
    system = args.system or args.domain
    out = Path(RUNS) / "abstain" / (f"{args.domain}_s{args.seed}_{args.protocol}_{args.unit}_{args.menu}" + (f"_gf{args.group_frames}" if args.group_frames else "")); out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"abstain_{args.domain}_{args.protocol}", str(out / "run.log"))
    critics, fusion, router, tau = load_system(system, args.seed, device)
    nets = (router, fusion, critics, tau); raw = RawAccess(args.domain)
    full = UnifiedPairedDataset(args.domain, label_mode="super", keep_empty=False)
    has_raw = any(r.get("ir16") for r in full.recs)
    m = build_menu(with_real=(args.menu == "both" and has_raw))
    scene = {r["id"]: r["scene"] for r in full.recs}
    if args.group_frames > 0:  # fine-grained groups: within a sequence (scene without the 250-frame block suffix) every N frames by id order
        from collections import defaultdict as _dd
        by_seq = _dd(list)
        for r in full.recs:
            by_seq["_".join(r["scene"].split("_")[:2])].append(r["id"])
        for sq, ids_ in by_seq.items():
            for pos, i in enumerate(sorted(ids_)):
                scene[i] = f"{sq}_f{pos // args.group_frames}"
    test_ids = (SPLITS / f"{args.domain}_test.txt").read_text(encoding="utf-8").split()
    if args.protocol == "exchangeable":
        groups = defaultdict(list)
        for i in test_ids:
            groups[scene[i]].append(i)
        names = sorted(groups, key=lambda g: hashlib.sha256(f"exch:{g}".encode()).hexdigest())
        half = len(test_ids) // 2; cal_ids, ev_ids, acc = [], [], 0
        for g in names:
            (cal_ids if acc < half else ev_ids).extend(groups[g]); acc += len(groups[g])
        log.info(f"exchangeable: calib={len(cal_ids)} eval={len(ev_ids)} groups={len(names)}")
    else:
        cal_ids = (SPLITS / f"{args.domain}_calib.txt").read_text(encoding="utf-8").split(); ev_ids = test_ids
    if args.limit:
        cal_ids, ev_ids = cal_ids[: args.limit], ev_ids[: args.limit]
    cal = collect(args.domain, UnifiedPairedDataset(args.domain, ids=cal_ids, label_mode="super", keep_empty=False), args.draws, device, nets, m, raw, log, "calib")
    s_cal = [r["score"] for r in cal]; l_cal = [r["loss"] for r in cal]; g_cal = [scene[r["image_id"]] for r in cal]
    calibration = {}
    for a in ALPHAS:
        lam, diag = (crc_threshold_grouped(s_cal, l_cal, g_cal, a) if args.unit == "group" else crc_threshold(s_cal, l_cal, a))
        calibration[str(a)] = diag
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    with open(out / "regions_calib.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cal[0].keys())); w.writeheader(); w.writerows(cal)
    te = collect(args.domain, UnifiedPairedDataset(args.domain, ids=ev_ids, label_mode="super", keep_empty=False), args.draws, device, nets, m, raw, log, "test")
    s_te = np.array([r["score"] for r in te]); l_te = np.array([r["loss"] for r in te]); g_te = [scene[r["image_id"]] for r in te]
    test_eval = {"n_test_regions": int(len(te)), "n_groups_calib": len(set(g_cal)), "n_groups_test": len(set(g_te))}
    for a in ALPHAS:
        lam = calibration[str(a)]["lambda_hat"]; lam = lam if lam is not None else np.inf
        if args.unit == "group":
            r = evaluate_threshold_grouped(s_te, l_te, g_te, lam); r["risk_bound_held"] = bool(r["group_risk"] <= a)
        else:
            r = evaluate_threshold(s_te, l_te, lam); r["risk_bound_held"] = bool(r["unconditional_risk"] <= a)
        r["alpha"] = a; test_eval[str(a)] = r
    (out / "test_eval.json").write_text(json.dumps(test_eval, indent=2), encoding="utf-8")
    cov, risk = risk_coverage_sweep(s_te, l_te)
    with open(out / "rc_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["coverage", "selective_risk"]); w.writerows(zip(cov.tolist(), risk.tolist()))
    with open(out / "regions_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(te[0].keys())); w.writeheader(); w.writerows(te)
    log.info(json.dumps({k: {kk: v[kk] for kk in ("risk_bound_held",)} for k, v in test_eval.items() if isinstance(v, dict)}))
    log.info("done")


if __name__ == "__main__":
    main()
