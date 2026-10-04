# -*- coding: utf-8 -*-
"""Offline fused images (clean) of train/val produced by the frozen joint fusion network, for training the
"three-modality mixed" evaluator (train_heldout_domain.py --fused-dir).
Usage: python c_precompute_fused.py --domain lynred_mds   Output: runs/fused_imgs/<domain>/<image_id>.png (uint8 grey)"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from c_eval import load_system  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", default="lynred_mds"); ap.add_argument("--splits", default="train,val")
    a = ap.parse_args(); device = "cuda"
    out = Path(RUNS) / "fused_imgs" / a.domain; out.mkdir(parents=True, exist_ok=True)
    critics, fusion, _, _ = load_system(a.domain, 0, device)
    for sp in a.splits.split(","):
        ids = (SPLITS / f"{a.domain}_{sp}.txt").read_text(encoding="utf-8").split()
        ds = UnifiedPairedDataset(a.domain, ids=ids, label_mode="super", keep_empty=True)
        ld = DataLoader(ds, batch_size=4, num_workers=4, collate_fn=collate_pairs)
        n = 0
        with torch.no_grad():
            for vis, ir, targets in ld:
                vt, it_ = vis.to(device), ir.to(device); vg = rgb_to_gray(vt); hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
                fused = fusion(vg, it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw)).clamp(0, 1)
                for k, t in enumerate(targets):
                    Image.fromarray((fused[k, 0].cpu().numpy() * 255).astype(np.uint8)).save(out / f"{t['image_id']}.png"); n += 1
        print(sp, n, "fused images", flush=True)


if __name__ == "__main__":
    main()
