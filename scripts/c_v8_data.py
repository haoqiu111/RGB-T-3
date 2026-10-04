# -*- coding: utf-8 -*-
"""Single source of the numbers reported in the paper (tables, figures and text read them from here).
"Full system" = label-free two-level gated inversion (tag inv_blind) + region selection; "selection only" = no inversion (tag tau0.04[_extra]).
Gains and NFR of the full system are always measured against always-fuse and best-single on the frames as emitted by the camera (q_fused / q_vis / q_ir of the
no-inversion run). The two runs are aligned box by box (same image_id, draw and box order); a misalignment raises."""
import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import _bootstrap  # noqa: F401,E402
from cdkit import INDEX, RUNS as _RUNS  # noqa: E402

RUNS = Path(_RUNS); CE = RUNS / "c_eval"; LYN = CE / "lynred_mds__lynred_mds"
EV = ("dom_retinanet", "dom_rtdetr", "dom_yolo"); EVN = {"dom_retinanet": "RetinaNet", "dom_rtdetr": "RT-DETR", "dom_yolo": "YOLO11m"}
INV = "inv_blind"; MARGIN = 0.02
BASE_TAG = {0: "tau0.04_extra", 1: "tau0.04", 2: "tau0.04"}
SYN = ["ir_stripe_nuc", "ir_gauss_noise", "ir_dead_pixels", "ir_region_drop", "ir_low_contrast", "misalignment", "vis_gauss_noise", "vis_region_drop", "vis_blur", "vis_overexposure", "vis_underexposure"]
REAL = ["real_agc_stale", "real_agc_histeq", "real_agc_plateau", "real_agc_minmax", "real_agc_gamma"]
KNAME = {"ir_stripe_nuc": "IR stripe NUC", "ir_gauss_noise": "IR gain noise", "ir_dead_pixels": "IR dead pixels", "ir_region_drop": "IR local dropout", "ir_low_contrast": "IR low contrast",
         "misalignment": "Registration shift", "vis_gauss_noise": "VIS noise", "vis_region_drop": "VIS local dropout", "vis_blur": "VIS blur", "vis_overexposure": "VIS over-exposure",
         "vis_underexposure": "VIS under-exposure", "real_agc_stale": "Real AGC: stale window", "real_agc_histeq": "Real AGC: histogram eq.", "real_agc_plateau": "Real AGC: plateau eq.",
         "real_agc_minmax": "Real AGC: min-max", "real_agc_gamma": "Real AGC: gamma"}


def summary(tag, seed=0, base=LYN, split="test"):
    return json.loads((base / f"{split}_both_s{seed}_{tag}" / "summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def rows(tag, ev, seed=0, base=str(LYN), split="test"):
    p = Path(base) / f"{split}_both_s{seed}_{tag}" / f"{ev}_regions.csv"
    rs = list(csv.DictReader(open(p, encoding="utf-8")))
    out = {"image_id": np.array([r["image_id"] for r in rs]), "kind": np.array([r["kind"] for r in rs]), "severity": np.array([r["severity"] for r in rs]),
           "box_idx": np.array([int(r["box_idx"]) for r in rs]), "action": np.array([int(r["action"]) for r in rs]),
           "inverted": np.array([int(r.get("inverted") or 0) for r in rs])}
    for k in rs[0]:
        if k.startswith("q_"):
            out[k[2:]] = np.array([float(r[k]) for r in rs])
    return out


def aligned(a, b):
    assert len(a["kind"]) == len(b["kind"]), (len(a["kind"]), len(b["kind"]))
    for k in ("image_id", "kind", "box_idx"):
        assert (a[k] == b[k]).all(), k


def masks(kind):
    stress = kind != "clean"; real = np.char.startswith(kind, "real")
    return {"clean": ~stress, "stress": stress, "stress_synthetic": stress & ~real, "stress_real": real}


@lru_cache(maxsize=None)
def pair(ev, seed=0):
    """Box-aligned arrays of (no-inversion run, gated run)."""
    b, f = rows(BASE_TAG[seed], ev, seed), rows(INV, ev, seed); aligned(b, f); return b, f


def block_means(ev, seed=0):
    """Block means (clean/stress/synthetic/real): as-emitted single modalities and always-fuse, selection only, full system, late fusion, and NFR."""
    b, f = pair(ev, seed); best = np.maximum(b["vis"], b["ir"]); out = {}
    for blk, m in masks(b["kind"]).items():
        o = {"n": int(m.sum()), "vis": b["vis"][m].mean(), "ir": b["ir"][m].mean(), "fused": b["fused"][m].mean(), "best": best[m].mean(),
             "sel_stitched": b["composed"][m].mean(), "sel_decision": b["decision"][m].mean(), "full_stitched": f["composed"][m].mean(), "full_decision": f["decision"][m].mean(),
             "full_fused": f["fused"][m].mean(), "full_ir": f["ir"][m].mean(), "inv_rate": f["inverted"][m].mean(), "switch_sel": (b["action"][m] != 0).mean(), "switch_full": (f["action"][m] != 0).mean()}
        for name, q in (("fused", b["fused"]), ("sel_stitched", b["composed"]), ("sel_decision", b["decision"]), ("full_stitched", f["composed"]), ("full_decision", f["decision"])):
            o["nfr_" + name] = float((q[m] < best[m] - MARGIN).mean())
        if "late" in b:
            o["late"] = b["late"][m].mean(); o["nfr_late"] = float((b["late"][m] < best[m] - MARGIN).mean())
        if "late" in f:
            o["full_late"] = f["late"][m].mean()
        out[blk] = {k: float(v) for k, v in o.items()}
    return out


def extra_means(tag, cand, ev, seed=0):
    """Block means and NFR of one extra candidate in a run (published models etc.), relative to the as-emitted best-single of that run."""
    r = rows(tag, ev, seed); best = np.maximum(r["vis"], r["ir"])
    return {blk: {"q": float(r[cand][m].mean()), "nfr": float((r[cand][m] < best[m] - MARGIN).mean())} for blk, m in masks(r["kind"]).items()}


def per_kind(ev, seed=0):
    b, f = pair(ev, seed); best = np.maximum(b["vis"], b["ir"]); out = {}
    for k in ["clean"] + SYN + REAL:
        m = b["kind"] == k
        out[k] = {"n": int(m.sum()), "vis": float(b["vis"][m].mean()), "ir": float(b["ir"][m].mean()), "fused": float(b["fused"][m].mean()), "best": float(best[m].mean()),
                  "sel": float(b["decision"][m].mean()), "full": float(f["decision"][m].mean()), "late": float(b["late"][m].mean()) if "late" in b else None,
                  "switch_full": float((f["action"][m] != 0).mean()), "switch_sel": float((b["action"][m] != 0).mean()), "inv_rate": float(f["inverted"][m].mean())}
    return out


def kind_means(tag, cand, ev, seed=0):
    r = rows(tag, ev, seed)
    return {k: float(r[cand][r["kind"] == k].mean()) for k in ["clean"] + SYN + REAL}


# ---------------- scene-group bootstrap (same construction as c_stats.py: groups of 50 consecutive frames within a sequence) ----------------
@lru_cache(maxsize=None)
def scene_groups(domain="lynred_mds", gf=50):
    idx = json.loads((Path(INDEX) / f"{domain}.json").read_text(encoding="utf-8")); by = defaultdict(list); scene = {}
    for r in idx["images"]:
        by["_".join(r["scene"].split("_")[:2])].append(r["id"])
    for sq, ids in by.items():
        for pos, i in enumerate(sorted(ids)):
            scene[i] = f"{sq}_f{pos // gf}"
    return scene


def grouped_ci(vals, groups, B=2000, seed=20260914):
    g = defaultdict(list)
    for v, k in zip(vals, groups):
        g[k].append(v)
    arrs = [np.array(x) for x in g.values()]; rng = np.random.default_rng(seed); st = []
    for _ in range(B):
        pick = rng.integers(0, len(arrs), len(arrs)); st.append(float(np.concatenate([arrs[i] for i in pick]).mean()))
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(np.concatenate(arrs).mean()), float(lo), float(hi), len(arrs)


def gains_ci(ev, seeds=(0, 1, 2)):
    """Gains with 95% CI (points), three seeds pooled: selection only = decision - fused; full system = gated decision - as-emitted fused."""
    sc = scene_groups(); d_sel, d_full, kinds, grp, per_seed = [], [], [], [], {"sel": defaultdict(list), "full": defaultdict(list)}
    for s in seeds:
        b, f = pair(ev, s); ds, df = b["decision"] - b["fused"], f["decision"] - b["fused"]
        d_sel.append(ds); d_full.append(df); kinds.append(b["kind"]); grp.append(np.array([sc[i] for i in b["image_id"]]))
        for blk, m in masks(b["kind"]).items():
            per_seed["sel"][blk].append(100 * ds[m].mean()); per_seed["full"][blk].append(100 * df[m].mean())
    d_sel, d_full, kinds, grp = map(np.concatenate, (d_sel, d_full, kinds, grp)); out = {}
    for blk, m in masks(kinds).items():
        o = {}
        for name, d in (("sel", d_sel), ("full", d_full)):
            p, lo, hi, ng = grouped_ci(100 * d[m], grp[m])
            o[name] = {"gain": p, "lo": lo, "hi": hi, "groups": ng, "seed_min": min(per_seed[name][blk]), "seed_max": max(per_seed[name][blk])}
        out[blk] = o
    return out
