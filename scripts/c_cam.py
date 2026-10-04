# -*- coding: utf-8 -*-
"""CAM figure: the three qualitative cases x [TIR as emitted | joint always-fuse | TarDAL | SeAFusion | ours composed];
heat map = foreground confidence of the in-domain hold-out RetinaNet classification head (absolute [0,1]).
Reads runs/figures_c/qual_selection_c.json ({"iid", "label"} per case). Output runs/figures_c/figC9_cam.{png,pdf}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap  # noqa: F401
from cdkit import RUNS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402
from cdkit.realdeg import apply_any, menu as build_menu  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from rgbta.models.router import compose  # noqa: E402
from c_eval import DomRetina, load_extra_fusers, load_system  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from det_cam_util import composite, det_cam  # noqa: E402
from precompute_region_labels import draw_rng  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402

OUT = Path(RUNS) / "figures_c"; device = "cuda"; DOM = "lynred_mds"


class _Wrap:
    def __init__(self, det):
        self.det = det


def main():
    critics, fusion, router, _ = load_system(DOM, 0, device); tau = 0.04; ev = DomRetina(DOM, device); fm = _Wrap(ev.m)
    fusers = load_extra_fusers(["tardal", "seafusion"], device, DOM); raw = RawAccess(DOM); m = build_menu(with_real=True)
    sel = json.loads((OUT / "qual_selection_c.json").read_text(encoding="utf-8"))
    KIND = {"Real AGC: stale window": "real_agc_stale", "Real AGC: histogram equalisation": "real_agc_histeq", "Synthetic IR stripe NUC": "ir_stripe_nuc"}
    rows = []
    for p in sel:
        iid, kind = p["iid"], KIND[p["label"]]
        ds = UnifiedPairedDataset(DOM, ids=[iid], label_mode="super", keep_empty=False); vis, ir, t = ds[0]
        v = vis.permute(1, 2, 0).numpy(); g = ir[0].numpy(); r16 = raw(iid); found = None
        for draw in range(1, 4):
            rng = draw_rng(f"ceval:{DOM}:{iid}", draw); k, sev = m[int(rng.integers(0, len(m)))]
            if k == kind:
                found = apply_any(k, sev, v, g, rng, r16); break
        vd, idg = found
        with torch.no_grad():
            vt = torch.from_numpy(np.ascontiguousarray(vd.transpose(2, 0, 1)))[None].to(device); it_ = torch.from_numpy(np.ascontiguousarray(idg[None][None])).to(device)
            vg = rgb_to_gray(vt); hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
            fused = fusion(vg, it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw))
            qhat = router(vg, it_); alt = qhat[:, 1:].max(dim=1, keepdim=True).values
            cell = torch.where((alt - qhat[:, :1]) >= tau, 1 + qhat[:, 1:].argmax(dim=1, keepdim=True), torch.zeros_like(alt, dtype=torch.long))
            composed, _ = compose(torch.zeros_like(qhat).scatter_(1, cell, 10.0), fused, vg, it_, hard=True)
            imgs = {"TIR as emitted": idg, "Joint always-fuse": fused[0, 0].clamp(0, 1).cpu().numpy(), "TarDAL": fusers["tardal"](vg, it_)[0, 0].clamp(0, 1).cpu().numpy(),
                    "SeAFusion": fusers["seafusion"](vg, it_)[0, 0].clamp(0, 1).cpu().numpy(), "Ours (composed)": composed[0, 0].clamp(0, 1).cpu().numpy()}
        rows.append((p["label"], {k: (im, det_cam(fm, im.astype(np.float32))) for k, im in imgs.items()}))
    fig, axs = plt.subplots(len(rows), 5, figsize=(13, 2.5 * len(rows)))
    for a, (label, d) in zip(axs, rows):
        for ax, (name, (im, heat)) in zip(a, d.items()):
            ax.imshow(composite(im, heat)); ax.set_axis_off(); ax.set_title(f"{label}\n{name}" if name == "TIR as emitted" else name, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "figC9_cam.pdf"); fig.savefig(OUT / "figC9_cam.png", dpi=300); print("saved figC9")


if __name__ == "__main__":
    main()
