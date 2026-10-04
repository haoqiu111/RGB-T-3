# -*- coding: utf-8 -*-
"""Real degradation menu: on top of the 11 synthetic degradations of `rgbta`, a family of real
AGC tone maps executed on the 16-bit raw counts.

real kinds (severity recorded as 0; parameters fixed by the method name):
  real_agc_histeq / real_agc_plateau / real_agc_gamma / real_agc_minmax / real_agc_stale
  applied to the infrared candidate: ir := resize(tone_map(raw16, method)); the visible image is unchanged.
  When raw16 is missing (non-16-bit domains) the real kinds are unavailable and the menu falls back to
  the synthetic one.
Single entry point apply_any(kind, sev, vis, ir, rng, raw16=None, ref_raw=None).
"""
import numpy as np
from PIL import Image

from cdkit.thermal16 import pct, tone_map
from rgbta.data.degradations import ALL_KINDS, apply_degradation

REAL_KINDS = ("real_agc_histeq", "real_agc_plateau", "real_agc_gamma", "real_agc_minmax", "real_agc_stale")
REAL_METHOD = {"real_agc_histeq": "histeq", "real_agc_plateau": "plateau:0.004", "real_agc_gamma": "gamma:0.6",
               "real_agc_minmax": "linear_minmax"}
SEVERITIES = (2, 3, 4)


def menu(with_real=True, real_weight=3):
    """List of (kind, sev); real kinds are repeated real_weight times to balance the three synthetic severities."""
    m = [(k, s) for k in ALL_KINDS for s in SEVERITIES]
    if with_real:
        m += [(k, 0) for k in REAL_KINDS for _ in range(real_weight)]
    return m


def _resize(a, size=(640, 512)):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def apply_any(kind, sev, vis, ir, rng, raw16=None, ref_raw=None):
    if kind.startswith("real_agc"):
        if raw16 is None:
            raise ValueError("real_agc kinds need the 16-bit raw counts")
        if kind == "real_agc_stale":
            # stale AGC window: percentile window of the reference (or this frame) shifted by about half its width
            ref = ref_raw if ref_raw is not None else raw16
            lo, hi = pct(ref, 1), pct(ref, 99)
            shift = float(rng.uniform(0.35, 0.65)) * (hi - lo) * float(rng.choice([-1, 1]))
            g = tone_map(raw16, f"fixed:{lo + shift},{hi + shift}")
        else:
            g = tone_map(raw16, REAL_METHOD[kind])
        return vis, _resize(g, (ir.shape[1], ir.shape[0]))
    return apply_degradation(kind, sev, vis, ir, rng)
