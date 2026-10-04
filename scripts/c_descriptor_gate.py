# -*- coding: utf-8 -*-
"""Descriptor-gated routing baseline (image-level "spectral reliability descriptor -> expert/modality gate"):
per modality a 7-dim descriptor = energy fractions of three radial frequency bands (low/mid/high), spectral entropy,
gradient energy, saturated-pixel fraction, mean local standard deviation;
gate = gradient-boosting classifier trained on the same counterfactual labels (argmax of the image-level mean q
over {FUSE, VIS, IR}); one candidate is chosen for the whole image.
Training: replay the degradations of the (image, draw) pairs in runs/router/lynred_mds_labels_train.csv and compute descriptors.
Usage: python c_descriptor_gate.py --domain lynred_mds   Output: runs/router/<domain>_dgate.joblib + report"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402
from cdkit.realdeg import apply_any, menu as build_menu  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from precompute_region_labels import draw_rng  # noqa: E402


def descriptor(g):
    """g: [H,W] float [0,1] grey image. Returns a 7-dim vector."""
    g = g.astype(np.float32); H, W = g.shape
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.mgrid[:H, :W]; r = np.sqrt(((yy - H / 2) / (H / 2)) ** 2 + ((xx - W / 2) / (W / 2)) ** 2)
    tot = F.sum() + 1e-9; low = F[r < 0.1].sum() / tot; mid = F[(r >= 0.1) & (r < 0.4)].sum() / tot; high = F[r >= 0.4].sum() / tot
    p = F.ravel() / tot; ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(p.size))
    gy, gx = np.gradient(g); grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    sat = float(((g > 0.98) | (g < 0.02)).mean())
    from scipy.ndimage import uniform_filter
    m = uniform_filter(g, 9); lstd = float(np.sqrt(np.maximum(uniform_filter(g * g, 9) - m * m, 0)).mean())
    return np.array([low, mid, high, ent, grad, sat, lstd], dtype=np.float32)


def pair_descriptor(v_np, i_np):
    vg = v_np.mean(axis=2); dv, di = descriptor(vg), descriptor(i_np)
    return np.concatenate([dv, di, dv - di])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", default="lynred_mds"); ap.add_argument("--max-images", type=int, default=1200)
    a = ap.parse_args(); out = Path(RUNS) / "router"
    lab = defaultdict(list)
    for r in csv.DictReader(open(out / f"{a.domain}_labels_train.csv", encoding="utf-8")):
        lab[(r["image_id"], int(r["draw"]))].append((float(r["qf"]), float(r["qv"]), float(r["qi"])))
    keys = sorted(lab); ids = sorted({k[0] for k in keys})[: a.max_images]; keep = {k for k in keys if k[0] in set(ids)}
    ds = UnifiedPairedDataset(a.domain, ids=ids, label_mode="super", keep_empty=False); raw = RawAccess(a.domain); m = build_menu(with_real=True)
    X, y, kinds = [], [], []
    for i in range(len(ds)):
        vis, ir, t = ds[i]; v = vis.permute(1, 2, 0).numpy(); g = ir[0].numpy(); r16 = raw(t["image_id"])
        for draw in range(3):
            if (t["image_id"], draw) not in keep:
                continue
            rng = draw_rng(f"{a.domain}:{t['image_id']}", draw)
            if draw == 0:
                kind, vd, idg = "clean", v, g
            else:
                kind, sev = m[int(rng.integers(0, len(m)))]
                if kind.startswith("real_agc") and r16 is None:
                    kind, sev = m[0]
                vd, idg = apply_any(kind, sev, v, g, rng, r16)
            q = np.mean(lab[(t["image_id"], draw)], axis=0); X.append(pair_descriptor(vd, idg)); y.append(int(np.argmax(q))); kinds.append(kind)
        if (i + 1) % 200 == 0:
            print(i + 1, "/", len(ds), flush=True)
    X, y = np.stack(X), np.array(y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=0).fit(X, y)
    joblib.dump({"clf": clf, "labels": ["fused", "vis", "ir"]}, out / f"{a.domain}_dgate.joblib")
    from sklearn.model_selection import cross_val_score
    acc = float(np.mean(cross_val_score(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=0), X, y, cv=5)))
    rep = {"n": int(len(y)), "label_prior": {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}, "cv_acc": round(acc, 4)}
    (out / f"{a.domain}_dgate_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8"); print(rep)


if __name__ == "__main__":
    main()
