# -*- coding: utf-8 -*-
"""16-bit raw thermal frame loading and the real AGC tone-map family.

The same raw counts run through different AGC algorithms give different 8-bit images; each one is
a frame the camera would actually emit rather than a pixel-level simulation.
Methods (all common industrial thermal-camera implementations):
  linear_minmax   full-frame min-max linear stretch
  linear_pct:p    linear stretch after clipping at the p and 100-p percentiles (p defaults to 1)
  histeq          global histogram equalisation
  plateau:k       plateau histogram equalisation (histogram clipped at k * total)
  gamma:g         gamma after percentile clipping
  fixed:lo,hi     fixed window (AGC locked / not updated with the scene = stale after drift)
  ema:alpha       temporal AGC: window statistics updated by an EMA (state kept by the caller)
Returns float32 in [0,1].
"""
import numpy as np
from PIL import Image


def load_ir16(path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.dtype != np.uint16:
        a = a.astype(np.uint16)
    return a.astype(np.float32)


def _stretch(a, lo, hi):
    return np.clip((a - lo) / max(float(hi - lo), 1.0), 0.0, 1.0).astype(np.float32)


def _valid(a: np.ndarray) -> np.ndarray:
    """Valid pixels: zeros are registration borders / invalid pixels and are excluded from all statistics."""
    v = a[a > 0]
    return v if v.size >= 64 else a.ravel()


def pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(_valid(a), p))


def tone_map(a: np.ndarray, method: str = "linear_pct:1", state: dict | None = None) -> np.ndarray:
    name, _, arg = method.partition(":")
    if name == "linear_minmax":
        v = _valid(a)
        return _stretch(a, v.min(), v.max())
    if name == "linear_pct":
        p = float(arg or 1)
        return _stretch(a, pct(a, p), pct(a, 100 - p))
    if name == "gamma":
        g = float(arg or 0.7)
        return np.power(_stretch(a, pct(a, 1), pct(a, 99)), g).astype(np.float32)
    if name == "fixed":
        lo, hi = (float(x) for x in arg.split(","))
        return _stretch(a, lo, hi)
    if name in ("histeq", "plateau"):
        lo, hi = pct(a, 0.1), pct(a, 99.9)
        q = np.clip(((a - lo) / max(hi - lo, 1.0)) * 4095, 0, 4095).astype(np.int32)
        hist = np.bincount(q.ravel(), minlength=4096).astype(np.float64)
        if name == "plateau":
            k = float(arg or 0.02)
            hist = np.minimum(hist, k * hist.sum())
        cdf = np.cumsum(hist)
        cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1.0)
        return cdf[q].astype(np.float32)
    if name == "ema":
        alpha = float(arg or 0.1)
        lo, hi = pct(a, 1), pct(a, 99)
        if state is None:
            state = {}
        if "lo" not in state:
            state["lo"], state["hi"] = lo, hi
        else:
            state["lo"] = (1 - alpha) * state["lo"] + alpha * lo
            state["hi"] = (1 - alpha) * state["hi"] + alpha * hi
        return _stretch(a, state["lo"], state["hi"])
    raise ValueError(f"unknown tone_map method: {method}")


def raw_stats(a: np.ndarray) -> dict:
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "p1": float(np.percentile(a, 1)), "p99": float(np.percentile(a, 99)),
            "col_bias_std": float(np.std(a.mean(axis=0) - a.mean()))}
