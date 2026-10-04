# -*- coding: utf-8 -*-
"""Deployment cost: single 640x512 frame, batch 1, CUDA-event timing (20 warm-up, 100 measured, median).
Components: fusion network / router / one detection per evaluator; systems: always-fuse (fusion + 1 detection),
ours stitched (fusion + router + 1 detection), ours decision-level (fusion + router + 3 detections), late fusion (2 detections).
The GPU must be idle. Output runs/c_cost/cost.json"""
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from cdkit import RUNS  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from c_eval import DomRetina, UltraEval, load_system  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402


def timeit(fn, n=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def main():
    device = "cuda"; out = Path(RUNS) / "c_cost"; out.mkdir(exist_ok=True)
    critics, fusion, router, _ = load_system("lynred_mds", 0, device)
    vt = torch.rand(1, 3, 512, 640, device=device); it3 = torch.rand(1, 3, 512, 640, device=device); it_ = it3[:, :1]
    vg = rgb_to_gray(vt); hw = (128, 160)
    evs = {"dom_retinanet": DomRetina("lynred_mds", device),
           "dom_rtdetr": UltraEval(Path(RUNS) / "heldout" / "lynred_mds_rtdetr" / "weights" / "best.pt", device, "rtdetr"),
           "dom_yolo": UltraEval(Path(RUNS) / "heldout" / "lynred_mds_yolo" / "weights" / "best.pt", device, "yolo")}
    with torch.no_grad():
        att = (det_attention(critics[0], vt, hw), det_attention(critics[1], it3, hw))
        t_fuse_deploy = timeit(lambda: fusion(vg, it_, None, None)) if _accepts_none(fusion, vg, it_) else None
        t_fuse_train = timeit(lambda: fusion(vg, it_, att[0], att[1]))
        t_router = timeit(lambda: router(vg, it_))
        rep = {"fusion_ms": t_fuse_deploy, "fusion_with_attention_ms": t_fuse_train, "router_ms": t_router,
               "params_fusion_M": sum(p.numel() for p in fusion.parameters()) / 1e6, "params_router_M": sum(p.numel() for p in router.parameters()) / 1e6, "detectors": {}}
        f_ms = t_fuse_deploy if t_fuse_deploy is not None else t_fuse_train
        for n, ev in evs.items():
            d = timeit(lambda: ev.predict(vt))
            rep["detectors"][n] = {"detect_ms": d, "always_fuse_ms": f_ms + d, "ours_stitched_ms": f_ms + t_router + d,
                                   "ours_decision_ms": f_ms + t_router + 3 * d, "late_fusion_ms": 2 * d, "single_modality_ms": d}
    (out / "cost.json").write_text(json.dumps(rep, indent=2), encoding="utf-8"); print(json.dumps(rep, indent=2))


def _accepts_none(fusion, vg, it_):
    try:
        with torch.no_grad():
            fusion(vg, it_, None, None)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
