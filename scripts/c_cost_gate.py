# -*- coding: utf-8 -*-
"""Per-frame cost of the gated inversion (CPU, numpy, 640x512 evaluation size).
Level one = monotone-fit residual; level two = camera map (FastEmulator: vectorised implementation whose output
matches the emulator_map used in the experiments) and deviation.
First checks the maximum absolute difference between FastEmulator and emulator_map frame by frame, then times
both; median over 60 frames. The machine must be idle.
Usage: python c_cost_gate.py   Output: runs/c_cost/cost_gate.json"""
import json
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402
from cdkit.ispgate import FastEmulator, emulator_map, monotone_residual  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402

ids = (SPLITS / "lynred_mds_test.txt").read_text(encoding="utf-8").split()[:70]
ds = UnifiedPairedDataset("lynred_mds", ids=ids, label_mode="super", keep_empty=False); raw = RawAccess("lynred_mds")
fast = FastEmulator(); frames = []
for i in range(len(ds)):
    _, ir, t = ds[i]; frames.append((ir[0].numpy(), raw(t["image_id"])))
diff = max(float(np.abs(fast(r16, (g.shape[1], g.shape[0])) - emulator_map(r16, (g.shape[1], g.shape[0]))).max()) for g, r16 in frames)
t1, t2, t2s = [], [], []
for i, (g, r16) in enumerate(frames):
    H, W = g.shape
    a = time.perf_counter(); monotone_residual(g, r16); b = time.perf_counter(); emu = fast(r16, (W, H)); float(np.abs(g - emu).mean()); c = time.perf_counter()
    emulator_map(r16, (W, H)); d = time.perf_counter()
    if i >= 10:
        t1.append(1000 * (b - a)); t2.append(1000 * (c - b)); t2s.append(1000 * (d - c))
rep = {"level1_residual_ms": float(np.median(t1)), "level2_camera_map_ms": float(np.median(t2)), "gate_total_ms": float(np.median(np.array(t1) + np.array(t2))),
       "level2_sklearn_reference_ms": float(np.median(t2s)), "fast_vs_reference_max_abs_diff": diff, "n": len(t1), "device": "CPU"}
(Path(RUNS) / "c_cost").mkdir(exist_ok=True)
(Path(RUNS) / "c_cost" / "cost_gate.json").write_text(json.dumps(rep, indent=2), encoding="utf-8"); print(rep)
