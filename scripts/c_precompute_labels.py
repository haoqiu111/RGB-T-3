# -*- coding: utf-8 -*-
"""Region-label precomputation for the router (ported from the companion project's precompute_region_labels.py);
menu = 11 synthetic degradations + 5 real AGC kinds.
Per image: clean + K draws; per-box evidence quality of the three candidates {fused, vis, ir} (mean of the two critics).
Usage: python c_precompute_labels.py --domain lynred_mds --splits train val --draws 2 [--no-real]
Output: runs/router/<domain>_labels_<split><variant>.csv (same schema as the companion project)"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import EXT, RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs, load_index  # noqa: E402
from cdkit.realdeg import apply_any, menu  # noqa: E402
from cdkit.thermal16 import load_ir16  # noqa: E402
from rgbta.models.fusion_net import FusionNet, fused_to_rgb, rgb_to_gray  # noqa: E402
from rgbta.utils.common import get_logger  # noqa: E402
from precompute_region_labels import cand_box_q, draw_rng  # noqa: E402
from train_critic_detector import build_model  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402


def read_ids(domain, tag):
    return (SPLITS / f"{domain}_{tag}.txt").read_text(encoding="utf-8").split()


def load_critic(domain, mod, device):
    ck = torch.load(Path(RUNS) / "critics" / f"{domain}_{mod}" / "best.pth", map_location="cpu", weights_only=False)
    m = build_model(ck["num_classes"]); m.load_state_dict(ck["model"]); return m.to(device).eval()


class RawAccess:
    """16-bit raw frame by image_id (crop applied)."""
    def __init__(self, domain):
        idx = load_index(domain); self.root = EXT / domain
        self.rec = {r["id"]: r for r in idx["images"]}
    def __call__(self, image_id):
        r = self.rec[image_id]
        if not r.get("ir16"):
            return None
        a = load_ir16(self.root / r["ir16"])
        if r.get("crop"):
            x0, y0, x1, y1 = r["crop"]; a = a[y0:y1, x0:x1]
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--draws", type=int, default=2); ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--variant", default="")
    args = ap.parse_args()
    device = "cuda"
    out_dir = Path(RUNS) / "router"; out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"precompute_{args.domain}", str(out_dir / f"precompute_{args.domain}.log"))
    critics = [load_critic(args.domain, "vis", device), load_critic(args.domain, "ir", device)]
    net = FusionNet("upstream").to(device).eval()
    net.load_state_dict(torch.load(Path(RUNS) / "fusion" / f"{args.domain}_joint" / "best.pth", map_location="cpu", weights_only=False)["model"])
    raw = RawAccess(args.domain)
    m = menu(with_real=not args.no_real)
    for split in args.splits:
        ds = UnifiedPairedDataset(args.domain, ids=read_ids(args.domain, split), label_mode="super", keep_empty=False)
        loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_pairs)
        out_csv = out_dir / f"{args.domain}_labels_{split}{args.variant}.csv"
        log.info(f"split={split} images={len(ds)} menu={len(m)} -> {out_csv.name}")
        with torch.no_grad(), open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "draw", "kind", "severity", "box_idx", "x1", "y1", "x2", "y2", "cls", "qf", "qv", "qi"])
            for bi, (vis, ir, targets) in enumerate(loader):
                t = targets[0]
                if len(t["boxes"]) == 0:
                    continue
                v_np = vis[0].permute(1, 2, 0).numpy(); i_np = ir[0, 0].numpy()
                raw16 = raw(t["image_id"])
                for draw in range(args.draws + 1):
                    rng = draw_rng(f"{args.domain}:{t['image_id']}", draw)
                    if draw == 0:
                        kind, sev, vd, idg = "clean", 0, v_np, i_np
                    else:
                        kind, sev = m[int(rng.integers(0, len(m)))]
                        if kind.startswith("real_agc") and raw16 is None:
                            kind, sev = m[0]
                        vd, idg = apply_any(kind, sev, v_np, i_np, rng, raw16)
                    vt = torch.from_numpy(np.ascontiguousarray(vd.transpose(2, 0, 1)))[None].to(device)
                    it_ = torch.from_numpy(np.ascontiguousarray(idg[None][None])).to(device)
                    hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
                    fused = net(rgb_to_gray(vt), it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw))
                    q_f = cand_box_q(critics, fused_to_rgb(fused), t); q_v = cand_box_q(critics, vt, t)
                    q_i = cand_box_q(critics, it_.repeat(1, 3, 1, 1), t)
                    if q_f is None:
                        continue
                    for k in range(len(q_f)):
                        b = t["boxes"][k].tolist()
                        w.writerow([t["image_id"], draw, kind, sev, k, round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1),
                                    int(t["labels"][k]), round(float(q_f[k]), 4), round(float(q_v[k]), 4), round(float(q_i[k]), 4)])
                if (bi + 1) % 200 == 0:
                    log.info(f"{split}: {bi+1}/{len(loader)}")
        log.info(f"split={split} done")


if __name__ == "__main__":
    main()
