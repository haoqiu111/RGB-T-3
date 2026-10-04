# -*- coding: utf-8 -*-
"""Single-modality critic for a new domain (Faster R-CNN R50-FPN), ported from the companion project's
train_critic_detector.py; data come from the unified index (super classes person/car/bicycle),
output goes to runs/critics/<domain>_<mod>/.
Usage: python c_train_critic.py --domain lynred_mds --modality vis --epochs 6"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from rgbta.utils.common import get_logger, set_seed  # noqa: E402
from train_critic_detector import build_model, eval_utility, pick_input, to_targets  # noqa: E402


def read_ids(domain, tag):
    return (SPLITS / f"{domain}_{tag}.txt").read_text(encoding="utf-8").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--modality", choices=["vis", "ir"], required=True)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.005); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed); device = "cuda"
    tag = f"{args.domain}_{args.modality}"
    out = Path(RUNS) / "critics" / tag; out.mkdir(parents=True, exist_ok=True)
    log = get_logger(tag, str(out / "train.log"))
    tr = UnifiedPairedDataset(args.domain, ids=read_ids(args.domain, "train"), label_mode="super", keep_empty=False)
    va = UnifiedPairedDataset(args.domain, ids=read_ids(args.domain, "val"), label_mode="super", keep_empty=False)
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6, collate_fn=collate_pairs, pin_memory=True,
                     drop_last=True, persistent_workers=True)
    lva = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_pairs, persistent_workers=True)
    n_cls = 3
    model = build_model(n_cls).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(ltr))
    scaler = torch.amp.GradScaler("cuda")
    log.info(f"train={len(tr)} val={len(va)} classes={n_cls}")
    best = -1
    for ep in range(args.epochs):
        model.train(); t0, running = time.time(), 0.0
        for it, (vis, ir, targets) in enumerate(ltr):
            imgs = pick_input(vis, ir, args.modality).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = sum(model(list(imgs), to_targets(targets, device)).values())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            running += float(loss)
            if (it + 1) % 100 == 0:
                log.info(f"ep{ep} it{it+1}/{len(ltr)} loss={running/100:.4f} {(it+1)*args.batch/(time.time()-t0):.1f} img/s"); running = 0.0
        u = eval_utility(model, lva, args.modality, device)
        log.info(f"ep{ep} val_utility={u:.4f}")
        st = {"model": model.state_dict(), "epoch": ep, "val_utility": u, "num_classes": n_cls, "modality": args.modality}
        torch.save(st, out / "last.pth")
        if u > best:
            best = u; torch.save(st, out / "best.pth")
    log.info(f"done. best val_utility={best:.4f}")


if __name__ == "__main__":
    main()
