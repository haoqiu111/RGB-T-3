# -*- coding: utf-8 -*-
"""Fusion network for a new domain (DetFusion-faithful architecture). --stage gp: gradient-preserving loss
baseline; --stage joint: detection attention + frozen-critic detection loss, warm-started from gp.
Ported from the companion project's train_fusion_baseline/joint.
Usage: python c_train_fusion.py --domain lynred_mds --stage gp --epochs 8
       python c_train_fusion.py --domain lynred_mds --stage joint --epochs 3
Output: runs/fusion/<domain>_{gp,joint}/best.pth"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from rgbta.losses.gp_loss import GPLoss  # noqa: E402
from rgbta.models.fusion_net import FusionNet, fused_to_rgb, rgb_to_gray  # noqa: E402
from rgbta.utils.common import get_logger, set_seed  # noqa: E402
from train_critic_detector import build_model  # noqa: E402
from train_fusion_joint import critic_loss_on, det_attention  # noqa: E402


def read_ids(domain, tag):
    return (SPLITS / f"{domain}_{tag}.txt").read_text(encoding="utf-8").split()


def load_critic(domain, mod, device):
    ck = torch.load(Path(RUNS) / "critics" / f"{domain}_{mod}" / "best.pth", map_location="cpu", weights_only=False)
    m = build_model(ck["num_classes"]); m.load_state_dict(ck["model"]); m.to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--stage", choices=["gp", "joint"], required=True)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0); ap.add_argument("--lam_det", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed); device = "cuda"
    batch = args.batch or (12 if args.stage == "gp" else 6)
    lr = args.lr or (1e-3 if args.stage == "gp" else 3e-4)
    out = Path(RUNS) / "fusion" / f"{args.domain}_{args.stage}"; out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"fusion_{args.domain}_{args.stage}", str(out / "train.log"))
    tr = UnifiedPairedDataset(args.domain, ids=read_ids(args.domain, "train"), label_mode="super", keep_empty=False)
    va = UnifiedPairedDataset(args.domain, ids=read_ids(args.domain, "val"), label_mode="super", keep_empty=False)
    ltr = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=6, collate_fn=collate_pairs, pin_memory=True,
                     drop_last=True, persistent_workers=True)
    lva = DataLoader(va, batch_size=batch, shuffle=False, num_workers=4, collate_fn=collate_pairs, persistent_workers=True)
    net = FusionNet("upstream").to(device)
    gp = GPLoss(alpha=1.0)
    joint = args.stage == "joint"
    if joint:
        critic_v, critic_i = load_critic(args.domain, "vis", device), load_critic(args.domain, "ir", device)
        ck = torch.load(Path(RUNS) / "fusion" / f"{args.domain}_gp" / "best.pth", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"]); log.info(f"warm start from {args.domain}_gp (val_loss={ck['val_loss']:.4f})")
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(ltr))
    log.info(f"train={len(tr)} val={len(va)} batch={batch} stage={args.stage}")

    def forward(vis, ir):
        vg = rgb_to_gray(vis)
        if joint:
            hw = (vg.shape[-2] // 4, vg.shape[-1] // 4)
            return vg, net(vg, ir, det_attention(critic_v, vis, hw), det_attention(critic_i, ir.repeat(1, 3, 1, 1), hw))
        return vg, net(vg, ir)

    best = float("inf")
    for ep in range(args.epochs):
        net.train(); t0, run_gp, run_det = time.time(), 0.0, 0.0
        for it, (vis, ir, targets) in enumerate(ltr):
            vis, ir = vis.to(device, non_blocking=True), ir.to(device, non_blocking=True)
            vg, fused = forward(vis, ir)
            tgt_dev = [{**t, "boxes": t["boxes"].to(device)} for t in targets]
            loss_gp = gp(fused, vg, ir, tgt_dev); loss = loss_gp
            if joint:
                f3 = fused_to_rgb(fused)
                loss_det = (critic_loss_on(critic_v, f3, targets, device) + critic_loss_on(critic_i, f3, targets, device)) / 2.0
                loss = loss_gp + args.lam_det * loss_det; run_det += float(loss_det)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run_gp += float(loss_gp)
            if (it + 1) % 100 == 0:
                log.info(f"ep{ep} it{it+1}/{len(ltr)} gp={run_gp/100:.4f} det={run_det/100:.4f} {(it+1)*batch/(time.time()-t0):.1f} img/s")
                run_gp, run_det = 0.0, 0.0
        net.eval(); vtot, n = 0.0, 0
        with torch.no_grad():
            for vis, ir, targets in lva:
                vis, ir = vis.to(device), ir.to(device)
                vg, fused = forward(vis, ir)
                vtot += float(gp(fused, vg, ir, [{**t, "boxes": t["boxes"].to(device)} for t in targets])); n += 1
        vtot /= max(n, 1); log.info(f"ep{ep} val_gp_loss={vtot:.4f}")
        st = {"model": net.state_dict(), "epoch": ep, "val_loss": vtot, "joint": joint}
        torch.save(st, out / "last.pth")
        if vtot < best:
            best = vtot; torch.save(st, out / "best.pth")
    log.info(f"done. best val_gp_loss={best:.4f}")


if __name__ == "__main__":
    main()
