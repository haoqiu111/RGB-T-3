# -*- coding: utf-8 -*-
"""Hold-out evaluator for a domain: RetinaNet R50-FPN v2 fine-tuned from COCO.

Trained on clean frames only (vis/ir alternating by sample parity = modality-agnostic evaluator), fully independent of the fusion network and the router,
frozen after training and used only for the final evaluation. Classes: --classes super (person/car/bicycle, as in the FLIR system)
or native (the native classes of the domain).
Usage: python train_heldout_domain.py --domain lynred_mds --classes super --epochs 8
Output: runs/heldout/<domain>_<classes>_retinanet/best.pth
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401,E402
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from rgbta.eval.utility import image_utility  # noqa: E402
from rgbta.utils.common import get_logger, set_seed  # noqa: E402
from train_heldout_retinanet import build_retinanet, mixed_input  # noqa: E402
import numpy as np
from PIL import Image


def mixed_input3(vis, ir, targets, idx, fused_dir):
    """vis / ir / fused rotate by sample ((idx + i) mod 3); fused frames are read by image_id from an offline directory."""
    out = []
    for i in range(vis.shape[0]):
        m = (idx + i) % 3
        if m == 0:
            out.append(vis[i])
        elif m == 1:
            out.append(ir[i].repeat(3, 1, 1))
        else:
            f = np.asarray(Image.open(Path(fused_dir) / f"{targets[i]['image_id']}.png"), dtype=np.float32) / 255.0
            out.append(torch.from_numpy(f)[None].repeat(3, 1, 1))
    return torch.stack(out)


def read_ids(domain, tag):
    p = SPLITS / f"{domain}_{tag}.txt"
    return p.read_text(encoding="utf-8").split() if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--classes", choices=["super", "native"], default="super")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ir-source", default="8bit")
    ap.add_argument("--max-train", type=int, default=0, help="limit the number of training images (large domains)")
    ap.add_argument("--fused-dir", default="", help="offline fused-frame directory; train/val rotate over vis/ir/fused")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    out = Path(RUNS) / "heldout" / (f"{args.domain}_{args.classes}_retinanet" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"heldout_{args.domain}", str(out / "train.log"))

    tr = UnifiedPairedDataset(args.domain, ids=read_ids(args.domain, "train"), label_mode=args.classes,
                              ir_source=args.ir_source, keep_empty=False)
    va = UnifiedPairedDataset(args.domain, ids=read_ids(args.domain, "val"), label_mode=args.classes,
                              ir_source=args.ir_source, keep_empty=False)
    if args.max_train and len(tr) > args.max_train:
        import hashlib
        tr.recs = sorted(tr.recs, key=lambda r: hashlib.sha256(f"{args.domain}:{r['id']}".encode()).hexdigest())[: args.max_train]
        tr.items = [r["id"] for r in tr.recs]
    n_cls = 3 if args.classes == "super" else len(tr.classes)
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6, collate_fn=collate_pairs,
                     pin_memory=True, drop_last=True, persistent_workers=True)
    lva = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_pairs,
                     persistent_workers=True)
    model = build_retinanet(n_cls).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(ltr))
    scaler = torch.amp.GradScaler("cuda")
    log.info(f"domain={args.domain} classes={args.classes}({n_cls}) train={len(tr)} val={len(va)} ir={args.ir_source}")
    best = -1
    for ep in range(args.epochs):
        model.train(); t0, running = time.time(), 0.0
        for it, (vis, ir, targets) in enumerate(ltr):
            imgs = (mixed_input3(vis, ir, targets, it, args.fused_dir) if args.fused_dir else mixed_input(vis, ir, it)).to(device, non_blocking=True)
            tgt = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device) + 1} for t in targets]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                losses = model(list(imgs), tgt); loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            running += float(loss)
            if (it + 1) % 100 == 0:
                log.info(f"ep{ep} it{it+1}/{len(ltr)} loss={running/100:.4f} {(it+1)*args.batch/(time.time()-t0):.1f} img/s")
                running = 0.0
        model.eval(); us = []
        with torch.no_grad():
            for bi, (vis, ir, targets) in enumerate(lva):
                if bi >= 60:
                    break
                preds = model(list((mixed_input3(vis, ir, targets, bi, args.fused_dir) if args.fused_dir else mixed_input(vis, ir, bi)).to(device)))
                for p, t in zip(preds, targets):
                    u = image_utility({"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(),
                                       "labels": p["labels"].cpu() - 1},
                                      {"boxes": t["boxes"], "labels": t["labels"]})
                    if u is not None:
                        us.append(u)
        u = sum(us) / max(len(us), 1)
        log.info(f"ep{ep} val_utility={u:.4f}")
        state = {"model": model.state_dict(), "epoch": ep, "val_utility": u, "num_classes": n_cls,
                 "classes": ["person", "car", "bicycle"] if args.classes == "super" else tr.classes}
        torch.save(state, out / "last.pth")
        if u > best:
            best = u; torch.save(state, out / "best.pth")
    log.info(f"best val_utility={best:.4f}")


if __name__ == "__main__":
    main()
