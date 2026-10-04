# -*- coding: utf-8 -*-
"""Gated AGC inversion ("invert, then fuse"): decide whether an 8-bit infrared frame is a
monotone remap of its 16-bit raw counts.

Principle: AGC / tone mapping (histogram equalisation, plateau equalisation, gamma, min-max,
locked window) is a per-pixel monotone function of the raw counts, so 8-bit = f(raw) can be
fitted monotonically with near-zero residual; stripes, noise, dead pixels and local dropout
are not functions of the raw counts and leave a large residual.
Gate passes -> re-execute the tone map on the raw counts (exact inversion); otherwise keep the frame.
Returns the residual RMS on the 8-bit [0,1] scale; the threshold is calibrated on clean frames
(see scripts/c_isp_gate_calib.py).
"""
import numpy as np
from PIL import Image

CANON = "linear_pct:1"
N_BINS = 256


def _resize(a, size):
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, Image.BILINEAR), dtype=np.float32)


def monotone_residual(ir8: np.ndarray, raw16: np.ndarray) -> float:
    """ir8: [H,W] float [0,1] at evaluation size; raw16: raw counts (any resolution, resized to ir8).
    Fits a monotone map f: raw -> ir8 (per raw-quantile bin means, forced non-decreasing by a
    cumulative maximum) and returns the residual RMS."""
    H, W = ir8.shape
    r = raw16.astype(np.float32)
    if r.shape != (H, W):
        r = _resize(r, (W, H))
    v = r > 0
    if v.sum() < 1024:
        v = np.ones_like(r, dtype=bool)
    rv, iv = r[v].ravel(), ir8[v].ravel()
    edges = np.quantile(rv, np.linspace(0, 1, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, rv, side="right") - 1, 0, N_BINS - 1)
    s = np.bincount(b, weights=iv, minlength=N_BINS); c = np.bincount(b, minlength=N_BINS).astype(np.float64)
    m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    # fill empty bins and enforce a non-decreasing map
    idx = np.arange(N_BINS); ok = ~np.isnan(m)
    m = np.interp(idx, idx[ok], m[ok]) if ok.sum() >= 2 else np.zeros(N_BINS)
    m = np.maximum.accumulate(m)
    return float(np.sqrt(np.mean((iv - m[b]) ** 2)))


def is_isp_remap(ir8: np.ndarray, raw16: np.ndarray, thr: float) -> bool:
    return monotone_residual(ir8, raw16) <= thr


_EMU = {}


def emulator_map(raw16, size):
    """Learned camera tone map (runs/nagc/lynred_mds/emulator2.joblib, curve regression):
    raw counts -> camera-style 8-bit frame at evaluation size."""
    if "regs" not in _EMU:
        import joblib, sys as _s
        from cdkit.paths import RUNS as _RUNS, SCRIPTS as _SCRIPTS
        _s.path.insert(0, str(_SCRIPTS))
        from c_train_agc_emulator2 import apply_curve, frame_feats
        d = joblib.load(_RUNS / "nagc" / "lynred_mds" / "emulator2.joblib")
        _EMU["regs"], _EMU["apply"], _EMU["feats"] = d["regs"], apply_curve, frame_feats
    a = np.asarray(raw16, dtype=np.float32); f, (lo, hi) = _EMU["feats"](a)
    knots = np.maximum.accumulate(np.clip(np.array([r.predict(f[None])[0] for r in _EMU["regs"]]), 0, 1)).astype(np.float32)
    g = _EMU["apply"](a, knots, lo, hi)
    W, H = size
    return g if g.shape == (H, W) else _resize(g, size)


class FastEmulator:
    """Equivalent fast implementation of emulator_map (used only for deployment timing; experiments
    use emulator_map): flattens all trees of the 33 HistGradientBoostingRegressors into arrays and
    evaluates the 33 knot values in one vectorised pass, avoiding the per-regressor sklearn predict
    overhead. Output matches emulator_map (checked frame by frame in c_cost_gate.py)."""

    def __init__(self):
        emulator_map(np.full((8, 8), 30000, dtype=np.float32), (8, 8))      # trigger loading
        regs = _EMU["regs"]; trees = []
        self.base = np.array([float(np.ravel(r._baseline_prediction)[0]) for r in regs]); self.owner = []
        for j, r in enumerate(regs):
            for it in r._predictors:
                trees.append(it[0].nodes); self.owner.append(j)
        n = max(len(t) for t in trees); T = len(trees)
        self.feat = np.zeros((T, n), np.int64); self.thr = np.zeros((T, n)); self.left = np.zeros((T, n), np.int64); self.right = np.zeros((T, n), np.int64)
        self.leaf = np.ones((T, n), bool); self.val = np.zeros((T, n))
        for i, t in enumerate(trees):
            m = len(t); self.feat[i, :m] = t["feature_idx"]; self.thr[i, :m] = t["num_threshold"]; self.left[i, :m] = t["left"]; self.right[i, :m] = t["right"]
            self.leaf[i, :m] = t["is_leaf"].astype(bool); self.val[i, :m] = t["value"]
        self.owner = np.array(self.owner); self.rows = np.arange(T); self.K = len(regs)

    def knots(self, f):
        x = np.asarray(f, dtype=np.float64); node = np.zeros(len(self.rows), np.int64)
        for _ in range(64):
            lf = self.leaf[self.rows, node]
            if lf.all():
                break
            go_left = x[self.feat[self.rows, node]] <= self.thr[self.rows, node]
            node = np.where(lf, node, np.where(go_left, self.left[self.rows, node], self.right[self.rows, node]))
        raw = self.base + np.bincount(self.owner, weights=self.val[self.rows, node], minlength=self.K)
        return np.maximum.accumulate(np.clip(raw, 0, 1)).astype(np.float32)

    def __call__(self, raw16, size):
        a = np.asarray(raw16, dtype=np.float32); f, (lo, hi) = _EMU["feats"](a)
        g = _EMU["apply"](a, self.knots(f), lo, hi); W, H = size
        return g if g.shape == (H, W) else _resize(g, size)


def maybe_invert(kind, ir8, raw16, mode="none", thr=0.02, canon="linear_pct", thr2=0.10):
    """Invert, then fuse. mode: none | gate | gate2 | gate2b | oracle.
    Protocol: synthetic degradations are sensor/registration properties that inversion cannot remove
    (they are present in the raw counts too), so the canonical AGC is only re-executed on the real AGC
    family (and on clean frames in gate mode); for synthetic kinds the gate decision is recorded but the
    frame is left unchanged. gate2b is label-free (deployment setting).
    Returns (ir8_out, inverted: bool, resid: float|None)."""
    if mode == "none" or raw16 is None:
        return ir8, False, None
    from cdkit.thermal16 import tone_map
    real = kind.startswith("real_agc"); clean = kind == "clean"
    resid = monotone_residual(ir8, raw16) if mode == "gate" else None
    if mode == "gate2b":
        # Blind two-level gate (never looks at the degradation label): any frame whose monotone residual is
        # <= thr and whose deviation from the camera's own map is >= thr2 is inverted to the emulator output.
        # The third return value carries the residual so trigger rates can be tabulated per kind.
        resid = monotone_residual(ir8, raw16)
        if resid <= thr:
            H, W = ir8.shape; emu = emulator_map(raw16, (W, H))
            if float(np.abs(ir8 - emu).mean()) >= thr2:
                return emu, True, resid
        return ir8, False, resid
    if mode == "oracle":
        fire = real
    elif mode == "gate2":
        # Two-level gate: monotone (an AGC-family frame) and deviating from the camera map (emulator) by more
        # than thr2 (the gain control really changed) -> invert
        resid = monotone_residual(ir8, raw16)
        if (real or clean) and resid <= thr:
            H, W = ir8.shape; emu = emulator_map(raw16, (W, H))
            fire = float(np.abs(ir8 - emu).mean()) >= thr2
            if fire:
                return emu, True, resid
        fire = False
    else:
        fire = (real or clean) and resid <= thr
    if not fire:
        return ir8, False, resid
    H, W = ir8.shape
    if canon == "emulator":
        return emulator_map(raw16, (W, H)), True, resid
    return _resize(tone_map(raw16, CANON), (W, H)), True, resid
