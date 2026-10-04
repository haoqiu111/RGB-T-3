# -*- coding: utf-8 -*-
"""Per-frame fusion time and parameter count of the published fusion models, same protocol as c_cost.py
(640x512, batch 1, CUDA-event timing, median). The GPU must be idle. Slow models (DRMF) are timed fewer times.
Parameter count = de-duplicated parameters of every nn.Module reachable from the fusion closure
(Text-IF includes its CLIP text encoder).
Usage: python c_cost_sota.py   Output: runs/c_cost/cost_sota.json"""
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from cdkit import RUNS  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from c_eval import load_extra_fusers, load_system  # noqa: E402

NAMES = ["tardal", "seafusion", "cddfuse", "emma", "textif", "textif_p", "controlfusion", "drmf"]


def timeit(fn, n, warm):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def closure_params(f):
    seen, total, visited = set(), 0, set()
    stack = [c.cell_contents for c in (getattr(f, "__closure__", None) or [])]
    while stack:
        o = stack.pop()
        if id(o) in visited:      # closures may reference each other; avoid infinite loops
            continue
        visited.add(id(o))
        if isinstance(o, torch.nn.Module):
            for p in o.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); total += p.numel()
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif callable(o) and getattr(o, "__closure__", None):
            stack.extend(c.cell_contents for c in o.__closure__)
    return total / 1e6


def main():
    device = "cuda"; out = Path(RUNS) / "c_cost"; out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    vt = torch.rand(1, 3, 512, 640, device=device); it_ = torch.rand(1, 1, 512, 640, device=device); vg = rgb_to_gray(vt)
    rep = {}
    with torch.no_grad():
        _, fusion, router, _ = load_system("lynred_mds", 0, device)
        rep["ours_fusion"] = {"ms": timeit(lambda: fusion(vg, it_, None, None), 100, 20), "params_M": sum(p.numel() for p in fusion.parameters()) / 1e6}
        rep["ours_router"] = {"ms": timeit(lambda: router(vg, it_), 100, 20), "params_M": sum(p.numel() for p in router.parameters()) / 1e6}
        del fusion, router
        for n in NAMES:
            try:
                f = load_extra_fusers([n], device, "lynred_mds")[n]
                fn = (lambda: f(vt, it_, "clean")) if getattr(f, "rgb", False) else (lambda: f(vg, it_))
                slow = n == "drmf"
                rep[n] = {"ms": timeit(fn, 8 if slow else 50, 2 if slow else 10), "params_M": round(closure_params(f), 3)}
            except Exception as e:  # one failing model must not stop the others
                rep[n] = {"error": repr(e)}
            print(n, rep[n], flush=True)
            del f; torch.cuda.empty_cache()
    (out / "cost_sota.json").write_text(json.dumps(rep, indent=2), encoding="utf-8"); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
