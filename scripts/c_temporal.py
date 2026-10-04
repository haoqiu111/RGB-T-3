# -*- coding: utf-8 -*-
"""Temporal stability: run the router frame by frame on the LYNRED stereo video streams (6 sequences, 8-bit visible +
thermal, registered pairs) and report the switch rate and frame-to-frame action consistency (fraction of cells whose
action is unchanged between consecutive frames). Two streams: clean (camera 8-bit output) and a real-AGC segment
(the middle third of every sequence gets its thermal frame replaced by a histeq / stale remap of the 16-bit counts).
Usage: python c_temporal.py --frames 240 --stride 2   Output: runs/c_temporal/report.json"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from cdkit import EXT, RUNS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, load_index  # noqa: E402
from cdkit.realdeg import apply_any  # noqa: E402
from cdkit.thermal16 import load_ir16  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from c_eval import load_system  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=240); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--tau", type=float, default=0.04)
    a = ap.parse_args(); device = "cuda"; dom = "lynred_stereo"
    _, _, router, _ = load_system("lynred_mds", 0, device); tau = a.tau
    idx = load_index(dom); root = EXT / dom
    seqs = sorted({r["scene"] for r in idx["images"]}); rep = {}; out = Path(RUNS) / "c_temporal"; out.mkdir(exist_ok=True)
    for seq in seqs:
        recs = sorted([r for r in idx["images"] if r["scene"] == seq], key=lambda r: r["id"])[: a.frames: a.stride]
        ds = UnifiedPairedDataset(dom, ids=[r["id"] for r in recs], label_mode="super", keep_empty=True)
        acts = {"clean": [], "agc": []}; rng = np.random.default_rng(0); n = len(ds); mid = (n // 3, 2 * n // 3)
        with torch.no_grad():
            for i in range(n):
                vis, ir, t = ds[i]; v = vis.permute(1, 2, 0).numpy(); g = ir[0].numpy()
                for stream in ("clean", "agc"):
                    gg = g
                    if stream == "agc" and mid[0] <= i < mid[1]:
                        r16 = load_ir16(root / recs[i]["ir16"]); kind = "real_agc_stale" if seq in seqs[::2] else "real_agc_histeq"
                        _, gg = apply_any(kind, 0, v, g, np.random.default_rng(7), r16)
                    vt = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))[None].to(device); it_ = torch.from_numpy(np.ascontiguousarray(gg[None][None])).to(device)
                    qhat = router(rgb_to_gray(vt), it_); alt = qhat[:, 1:].max(dim=1, keepdim=True).values
                    cell = torch.where((alt - qhat[:, :1]) >= tau, 1 + qhat[:, 1:].argmax(dim=1, keepdim=True), torch.zeros_like(alt, dtype=torch.long))
                    acts[stream].append(cell[0, 0].cpu().numpy())
        r = {}
        for stream, al in acts.items():
            al = np.stack(al); sw = float((al != 0).mean()); cons = float(np.mean([(al[j] == al[j - 1]).mean() for j in range(1, len(al))]))
            seg = slice(mid[0], mid[1]); sw_mid = float((al[seg] != 0).mean()); sw_out = float((np.concatenate([al[:mid[0]], al[mid[1]:]]) != 0).mean())
            r[stream] = {"switch_rate": round(sw, 4), "frame_consistency": round(cons, 4), "switch_mid": round(sw_mid, 4), "switch_outside": round(sw_out, 4), "n_frames": int(len(al))}
        rep[seq] = r; print(seq, r, flush=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    agg = {s: {k: round(float(np.mean([rep[q][s][k] for q in rep])), 4) for k in ("switch_rate", "frame_consistency", "switch_mid", "switch_outside")} for s in ("clean", "agc")}
    print("MEAN", agg); (out / "summary.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
