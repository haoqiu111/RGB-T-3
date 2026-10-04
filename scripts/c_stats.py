# -*- coding: utf-8 -*-
"""Scene-group bootstrap confidence intervals (B=2000) + clean non-inferiority test (delta=0.02).
Resamples the *_regions.csv of runs/c_eval/<system>__<domain>/<split>_both_s<seed>_<tag> by scene group and reports
95% CIs of decision-fused, composed-fused and late-decision (if present), clean/stress/synthetic/real separately,
per seed and pooled (pooled = rows of all seeds concatenated, then resampled by scene group).
Usage: python c_stats.py --system lynred_mds --domain lynred_mds --tag tau0.04 --seeds 0,1,2
Output: runs/c_eval/<system>__<domain>/bootstrap_<tag>.json and .md"""
import argparse
import sys
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cdkit.paths import REGISTRY, RUNS  # noqa: E402

RUNS = Path(RUNS); REG = Path(REGISTRY) / "index"
B = 2000; DELTA = 0.02
rng = np.random.default_rng(20260914)


def grouped_ci(vals, groups):
    g = defaultdict(list)
    for v, k in zip(vals, groups):
        g[k].append(v)
    arrs = [np.array(x) for x in g.values()]
    if not arrs:
        return None
    point = float(np.concatenate(arrs).mean()); stats = []
    for _ in range(B):
        pick = rng.integers(0, len(arrs), len(arrs))
        stats.append(float(np.concatenate([arrs[i] for i in pick]).mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_groups": len(arrs), "n_boxes": int(sum(len(a) for a in arrs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="lynred_mds"); ap.add_argument("--domain", default="lynred_mds")
    ap.add_argument("--tag", default="tau0.04"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--split", default="test")
    ap.add_argument("--evaluators", default="dom_retinanet,dom_rtdetr,dom_yolo")
    ap.add_argument("--group-frames", type=int, default=0, help=">0: fine-grained groups = every N frames in id order within a sequence (same as c_calibrate)")
    args = ap.parse_args()
    idx = json.loads((REG / f"{args.domain}.json").read_text(encoding="utf-8"))
    scene = {r["id"]: r.get("scene", r["id"]) for r in idx["images"]}
    if args.group_frames > 0:
        by_seq = defaultdict(list)
        for r in idx["images"]:
            by_seq["_".join(r["scene"].split("_")[:2])].append(r["id"])
        for sq, ids_ in by_seq.items():
            for pos, i in enumerate(sorted(ids_)):
                scene[i] = f"{sq}_f{pos // args.group_frames}"
    base = RUNS / "c_eval" / f"{args.system}__{args.domain}"
    seeds = [int(s) for s in args.seeds.split(",")]
    report = {}
    for ev in args.evaluators.split(","):
        per_seed = {}
        for s in seeds:
            p = base / (f"{args.split}_both_s{s}" + (f"_{args.tag}" if args.tag else "")) / f"{ev}_regions.csv"
            if not p.exists():
                continue
            per_seed[s] = list(csv.DictReader(open(p, encoding="utf-8")))
        if not per_seed:
            continue
        pools = {f"seed{s}": rows for s, rows in per_seed.items()}
        if len(per_seed) > 1:
            pools["pooled"] = [r for rows in per_seed.values() for r in rows]
        for pname, rows in pools.items():
            for cond, pred in (("clean", lambda r: r["kind"] == "clean"), ("stress", lambda r: r["kind"] != "clean"),
                               ("stress_synthetic", lambda r: r["kind"] != "clean" and not r["kind"].startswith("real")),
                               ("stress_real", lambda r: r["kind"].startswith("real"))):
                sub = [r for r in rows if pred(r)]
                if not sub:
                    continue
                grp = [scene[r["image_id"]] for r in sub]
                out = {"decision_minus_fused": grouped_ci([float(r["q_decision"]) - float(r["q_fused"]) for r in sub], grp),
                       "composed_minus_fused": grouped_ci([float(r["q_composed"]) - float(r["q_fused"]) for r in sub], grp),
                       "decision_minus_bestsingle": grouped_ci([float(r["q_decision"]) - max(float(r["q_vis"]), float(r["q_ir"])) for r in sub], grp)}
                if "q_late" in sub[0]:
                    out["late_minus_decision"] = grouped_ci([float(r["q_late"]) - float(r["q_decision"]) for r in sub], grp)
                    out["late_minus_bestsingle"] = grouped_ci([float(r["q_late"]) - max(float(r["q_vis"]), float(r["q_ir"])) for r in sub], grp)
                if cond == "clean":
                    out["noninferior"] = bool(out["decision_minus_fused"]["ci95"][0] > -DELTA); out["delta"] = DELTA
                report[f"{ev}/{pname}/{cond}"] = out
    tag = (args.tag or "default") + (f"_gf{args.group_frames}" if args.group_frames else "")
    (base / f"bootstrap_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# bootstrap CI ({args.system} on {args.domain}, tag={tag}, B={B}, scene groups)", "",
             "| evaluator | pool | condition | decision−fused | 95% CI | composed−fused | late−decision | groups | non-inferior |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in report.items():
        ev, pn, cond = k.split("/"); d = v["decision_minus_fused"]; c = v["composed_minus_fused"]; l = v.get("late_minus_decision")
        lines.append(f"| {ev} | {pn} | {cond} | {d['point']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {c['point']:+.4f} | "
                     f"{(f'{l[chr(112)+chr(111)+chr(105)+chr(110)+chr(116)]:+.4f}' if l else '–')} | {d['n_groups']} | {v.get('noninferior', '')} |")
    (base / f"bootstrap_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
