# -*- coding: utf-8 -*-
"""Learned camera tone map (AGC emulator, curve regression): for every frame the camera's 8-bit output is
fitted monotonically against the raw counts, giving the curve value at K=32 raw-quantile knots;
features = 64-bin histogram of the raw counts (normalised per frame to [p0.1, p99.9]) + absolute level
(p0.1 / p50 / p99.9 / mean divided by 65535); one gradient-boosting regressor per knot.
Usage: python c_train_agc_emulator2.py   Output: runs/nagc/lynred_mds/emulator2.joblib + emulator2_report.json"""
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor

import _bootstrap  # noqa: F401
from cdkit import EXT, RUNS, SPLITS  # noqa: E402
from cdkit.index import load_index  # noqa: E402
from cdkit.thermal16 import load_ir16, tone_map  # noqa: E402

K, NB = 32, 64


def frame_feats(a):
    v = a[a > 0] if (a > 0).sum() >= 64 else a.ravel()
    lo, hi = np.percentile(v, 0.1), np.percentile(v, 99.9); x = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1)
    h = np.histogram(x, bins=NB, range=(0, 1))[0].astype(np.float32); h /= max(h.sum(), 1)
    return np.concatenate([h, [lo / 65535, np.median(v) / 65535, hi / 65535, v.mean() / 65535]]).astype(np.float32), (lo, hi)


def fit_curve(a, g8, lo, hi):
    """Monotone curve of the camera 8-bit output against the normalised raw counts, at K+1 equally spaced knots."""
    x = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1).ravel(); y = g8.ravel()
    b = np.clip((x * K).astype(int), 0, K - 1); s = np.bincount(b, weights=y, minlength=K); c = np.bincount(b, minlength=K)
    m = np.where(c > 0, s / np.maximum(c, 1), np.nan); idx = np.arange(K); ok = ~np.isnan(m)
    m = np.interp(idx, idx[ok], m[ok]); m = np.maximum.accumulate(m)
    return np.concatenate([[m[0]], m]).astype(np.float32)  # K+1 knots: knot 0 takes the first bin value


def apply_curve(a, knots, lo, hi):
    x = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1); pos = x * K; i = np.clip(pos.astype(int), 0, K - 1); f = pos - i
    return (knots[i] + f * (knots[i + 1] - knots[i])).astype(np.float32)


def load_pairs(split, limit=0):
    idx = load_index("lynred_mds"); root = EXT / "lynred_mds"; ids = set((SPLITS / f"lynred_mds_{split}.txt").read_text(encoding="utf-8").split())
    recs = [r for r in idx["images"] if r["id"] in ids and r.get("ir16") and r.get("ir")][: limit or None]
    X, Y, meta = [], [], []
    for r in recs:
        a = load_ir16(root / r["ir16"]); g8 = np.asarray(Image.open(root / r["ir"]).convert("L"), dtype=np.float32) / 255.0
        if r.get("crop"):
            x0, y0, x1, y1 = r["crop"]; a = a[y0:y1, x0:x1]; g8 = g8[y0:y1, x0:x1]
        if g8.shape != a.shape:
            g8 = np.asarray(Image.fromarray(g8).resize((a.shape[1], a.shape[0]), Image.BILINEAR), dtype=np.float32)
        f, (lo, hi) = frame_feats(a); X.append(f); Y.append(fit_curve(a, g8, lo, hi)); meta.append((a, g8, lo, hi))
    return np.stack(X), np.stack(Y), meta


def main():
    Xtr, Ytr, _ = load_pairs("train"); Xva, Yva, mva = load_pairs("val")
    regs = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0).fit(Xtr, Ytr[:, j]) for j in range(K + 1)]
    P = np.stack([r.predict(Xva) for r in regs], axis=1); P = np.maximum.accumulate(np.clip(P, 0, 1), axis=1)
    e_emu, e_fit, e_can = [], [], []
    for (a, g8, lo, hi), p, y in zip(mva, P, Yva):
        e_emu.append(np.abs(apply_curve(a, p, lo, hi) - g8).mean()); e_fit.append(np.abs(apply_curve(a, y, lo, hi) - g8).mean()); e_can.append(np.abs(tone_map(a, "linear_pct:1") - g8).mean())
    rep = {"val_l1_emulator2": float(np.mean(e_emu)), "val_l1_perframe_monotone_fit": float(np.mean(e_fit)), "val_l1_canonical": float(np.mean(e_can)), "n_train": int(len(Xtr)), "n_val": int(len(Xva))}
    (Path(RUNS) / "nagc" / "lynred_mds").mkdir(parents=True, exist_ok=True)
    joblib.dump({"regs": regs, "K": K}, Path(RUNS) / "nagc" / "lynred_mds" / "emulator2.joblib")
    (Path(RUNS) / "nagc" / "lynred_mds" / "emulator2_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8"); print(rep)


if __name__ == "__main__":
    main()
