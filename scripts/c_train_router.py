# -*- coding: utf-8 -*-
"""Router v2 (quality regression) trained on a new domain, ported from the companion project's train_router_v2.py;
the degradation replay menu = 11 synthetic + 5 real AGC kinds (identical to c_precompute_labels).
Usage: python c_train_router.py --domain lynred_mds --epochs 8 --seed 0 [--no-real]
Output: runs/router/<domain>_router_v2<variant>_s<seed>/best.pth"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402
from cdkit.realdeg import apply_any, menu  # noqa: E402
from rgbta.models.router import RouterNetV2  # noqa: E402
from rgbta.utils.common import get_logger, set_seed  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from precompute_region_labels import draw_rng  # noqa: E402
from train_router import collate, load_label_csv  # noqa: E402
from train_router_v2 import run_epoch  # noqa: E402


def read_ids(domain, tag):
    return (SPLITS / f"{domain}_{tag}.txt").read_text(encoding="utf-8").split()


class DomainRouterDataset(Dataset):
    def __init__(self, domain, split, variant="", with_real=True):
        self.domain = domain
        self.base = UnifiedPairedDataset(domain, ids=read_ids(domain, split), label_mode="super", keep_empty=False)
        self.index = {s: i for i, s in enumerate(self.base.items)}
        self.samples = load_label_csv(Path(RUNS) / "router" / f"{domain}_labels_{split}{variant}.csv")
        self.keys = [k for k in self.samples if k[0] in self.index]
        self.menu = menu(with_real=with_real); self.raw = RawAccess(domain)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        image_id, draw = self.keys[i]
        s = self.samples[(image_id, draw)]
        vis, ir, _ = self.base[self.index[image_id]]
        v_np = vis.permute(1, 2, 0).numpy(); i_np = ir[0].numpy()
        rng = draw_rng(f"{self.domain}:{image_id}", draw)
        if s["kind"] != "clean":
            kind, sev = self.menu[int(rng.integers(0, len(self.menu)))]
            raw16 = self.raw(image_id) if kind.startswith("real_agc") else None
            if kind.startswith("real_agc") and raw16 is None:
                kind, sev = self.menu[0]
            assert kind == s["kind"] and sev == s["sev"], f"degradation replay mismatch {kind}/{sev} vs {s['kind']}/{s['sev']}"
            v_np, i_np = apply_any(kind, sev, v_np, i_np, rng, raw16)
        vg = (torch.from_numpy(v_np) @ torch.tensor([0.299, 0.587, 0.114]))[None]
        return (vg.float(), torch.from_numpy(np.ascontiguousarray(i_np[None])).float(),
                torch.tensor(s["boxes"], dtype=torch.float32).reshape(-1, 4),
                torch.tensor(s["action"], dtype=torch.long), torch.tensor(s["best_q"], dtype=torch.float32),
                torch.tensor(s["qvec"], dtype=torch.float32).reshape(-1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--variant", default="")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-real", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed); device = "cuda"
    out = Path(RUNS) / "router" / f"{args.domain}_router_v2{args.variant}_s{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"router_{args.domain}_s{args.seed}", str(out / "train.log"))
    tr = DomainRouterDataset(args.domain, "train", args.variant, not args.no_real)
    va = DomainRouterDataset(args.domain, "val", args.variant, not args.no_real)
    log.info(f"samples: train={len(tr)} val={len(va)}")
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6, collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=True)
    lva = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate, persistent_workers=True)
    model = RouterNetV2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1e9
    for ep in range(args.epochs):
        run_epoch(model, ltr, device, opt, log, f"ep{ep}")
        l1, gain = run_epoch(model, lva, device)
        log.info(f"ep{ep} val l1={l1:.4f} decision_gain={gain:+.4f}")
        torch.save({"model": model.state_dict(), "epoch": ep, "val_gain": gain}, out / "last.pth")
        if gain > best:
            best = gain; torch.save({"model": model.state_dict(), "epoch": ep, "val_gain": gain}, out / "best.pth")
    log.info(f"done. best val decision_gain={best:+.4f}")


if __name__ == "__main__":
    main()
