# -*- coding: utf-8 -*-
"""Late fusion as a fourth action (frame-level policy, offline simulation).
Input: a regions.csv containing q_late (an evaluation run with --extra late). The routing signal per frame and draw is the
fraction of switched regions in that frame (approximated by box-level actions).
Policy P(rho): frames with switch fraction >= rho go to late fusion (2 detections, q_late), otherwise to the stitched image
(1 detection, q_composed); variant P_dec(rho): otherwise decision-level (3 detections, q_decision).
Outputs quality versus mean number of detection passes, clean/stress/real separately.
Usage: python c_late_policy.py --run test_both_s0_tau0.04_extra
Output: runs/c_late_policy/<run>/curve.json + curve.md"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cdkit.paths import RUNS  # noqa: E402

RUNS = Path(RUNS)
RHOS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="test_both_s0_tau0.04_extra"); ap.add_argument("--system", default="lynred_mds__lynred_mds")
    a = ap.parse_args(); base = RUNS / "c_eval" / a.system / a.run; out = RUNS / "c_late_policy" / a.run; out.mkdir(parents=True, exist_ok=True)
    rep = {}; lines = ["| evaluator | block | ρ | frames→late | mean passes | quality (stitch/late policy) | quality (decision/late policy) | always-fuse | ours decision | late |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ev in ("dom_retinanet", "dom_rtdetr", "dom_yolo"):
        p = base / f"{ev}_regions.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        # group the boxes of one draw by (image_id, kind, severity)
        fr = defaultdict(list)
        for r in rows:
            fr[(r["image_id"], r["kind"], r["severity"])].append(r)
        blocks = {"clean": lambda k: k == "clean", "stress": lambda k: k != "clean", "stress_real": lambda k: k.startswith("real"), "stress_synthetic": lambda k: k != "clean" and not k.startswith("real")}
        rep[ev] = {}
        for bname, pred in blocks.items():
            frames = [v for k, v in fr.items() if pred(k[1])]
            if not frames:
                continue
            sw = np.array([np.mean([float(r["action"]) != 0 for r in v]) for v in frames])
            qL = np.array([np.mean([float(r["q_late"]) for r in v]) for v in frames]); nb = np.array([len(v) for v in frames])
            qC = np.array([np.mean([float(r["q_composed"]) for r in v]) for v in frames]); qD = np.array([np.mean([float(r["q_decision"]) for r in v]) for v in frames])
            qF = np.array([np.mean([float(r["q_fused"]) for r in v]) for v in frames])
            wavg = lambda q: float((q * nb).sum() / nb.sum())
            rep[ev][bname] = {"always_fuse": wavg(qF), "decision": wavg(qD), "late": wavg(qL), "stitched": wavg(qC), "policy": []}
            for rho in RHOS:
                late = sw >= rho if rho <= 1 else np.zeros_like(sw, dtype=bool)
                q1 = np.where(late, qL, qC); q3 = np.where(late, qL, qD)
                passes1 = float(np.mean(np.where(late, 2, 1))); passes3 = float(np.mean(np.where(late, 2, 3)))
                e = {"rho": rho, "frac_late": float(late.mean()), "passes_stitch": passes1, "q_stitch_policy": wavg(q1), "passes_decision": passes3, "q_decision_policy": wavg(q3)}
                rep[ev][bname]["policy"].append(e)
                lines.append(f"| {ev} | {bname} | {rho if rho <= 1 else 'never'} | {100 * e['frac_late']:.0f}% | {passes1:.2f} | {e['q_stitch_policy']:.3f} | {e['q_decision_policy']:.3f} (passes {passes3:.2f}) | {wavg(qF):.3f} | {wavg(qD):.3f} | {wavg(qL):.3f} |")
    (out / "curve.json").write_text(json.dumps(rep, indent=1), encoding="utf-8"); (out / "curve.md").write_text("\n".join(lines), encoding="utf-8")
    import sys; sys.stdout.reconfigure(encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
