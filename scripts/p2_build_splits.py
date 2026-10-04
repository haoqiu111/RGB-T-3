# -*- coding: utf-8 -*-
"""Frozen splits: the official test split is untouched; the official train split is cut 80/10/10 by scene group into
train/val/calib, with the group order fixed by SHA-256(f"{domain}:{seed}:{scene}") and therefore independent of the environment.

Domains without an official test split (utokyo: all; m3fd: existing train/eval) are cut 50/50 by scene group into train-half /
eval-half. Stream domains (lynred_stereo / freiburg train) are not split.
Output: registry/splits/<domain>_{train,val,calib,test,eval}.txt (existing files are never overwritten).
"""
import argparse
import hashlib
import sys
from collections import defaultdict

import _bootstrap  # noqa: F401,E402
from cdkit import INDEX, SPLITS  # noqa: E402
from cdkit.index import load_index  # noqa: E402

SEED = "cd2026"


def order_groups(domain, groups):
    return sorted(groups, key=lambda g: hashlib.sha256(f"{domain}:{SEED}:{g}".encode()).hexdigest())


def cut(domain, groups, fracs):
    """Accumulate groups into the given fractions; returns one id list per fraction."""
    total = sum(len(v) for v in groups.values())
    bounds, acc = [], 0.0
    for f in fracs:
        acc += f; bounds.append(acc * total)
    parts = [[] for _ in fracs]
    seen = 0
    for g in order_groups(domain, groups):
        k = min(i for i, b in enumerate(bounds) if seen < b) if seen < bounds[-1] else len(fracs) - 1
        parts[k].extend(groups[g]); seen += len(groups[g])
    return parts


def write(name, ids):
    p = SPLITS / f"{name}.txt"
    if p.exists():
        print(f"  {p.name} is frozen, not overwritten ({len(p.read_text().split())} ids)"); return
    p.write_text("\n".join(sorted(ids)), encoding="utf-8")
    print(f"  {p.name}: {len(ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="")
    args = ap.parse_args()
    SPLITS.mkdir(parents=True, exist_ok=True)
    doms = [d for d in args.domains.split(",") if d] or sorted(p.stem for p in INDEX.glob("*.json"))
    for d in doms:
        idx = load_index(d)
        by_split = defaultdict(lambda: defaultdict(list))
        for r in idx["images"]:
            by_split[r["split"]][r["scene"]].append(r["id"])
        print(f"[{d}] splits={ {k: sum(len(v) for v in g.values()) for k, g in by_split.items()} }")
        if d in ("flir_align", "llvip", "m3fd"):
            print("  split taken from the companion project, skipped"); continue
        if "stream" in by_split and len(by_split) == 1:
            print("  stream domain, not split"); continue
        if "train" in by_split:
            tr, va, ca = cut(d, by_split["train"], (0.8, 0.1, 0.1))
            write(f"{d}_train", tr); write(f"{d}_val", va); write(f"{d}_calib", ca)
            # official test set: 'test' if present, otherwise the official 'val' (rgbt_droneperson)
            test_tag = "test" if "test" in by_split else ("val" if "val" in by_split else None)
            if test_tag:
                write(f"{d}_test", [i for g in by_split[test_tag].values() for i in g])
        elif "all" in by_split:
            a, b = cut(d, by_split["all"], (0.5, 0.5))
            write(f"{d}_train", a); write(f"{d}_eval", b)
        else:
            print(f"  unrecognized split structure: {list(by_split)}")


if __name__ == "__main__":
    main()
