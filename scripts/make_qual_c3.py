# -*- coding: utf-8 -*-
"""Qualitative figures and detection-confidence CAM figures (two scene sets, each its own figure).
  1) "Ours" = full system: label-free two-level gated inversion (gate2b), then fusion and selection; published models consume the frame as emitted.
  2) The Ours column shows the composed image; the region action map is a small inset (grey FUSE / blue VIS / orange TIR).
  3) Set A ends with an inversion row: VIS | TIR as emitted | TIR after gated inversion | Joint fusion as emitted | Joint fusion after inversion | Ours.
  set A (figC7_qualitative_v8 / figC9_cam_v8): stale window, histogram eq., stripe NUC + the inversion row
  set B（figC12_qualitative2_v8 / figC13_cam2_v8）：min-max、gamma、dead pixels
Scenes: the two synthetic rows reuse scene_pick_v7.json; real-AGC rows maximize the per-image RT-DETR mean of "full-system stitched image - max(published models)"
with at least 3 objects and not below the emitted TIR; the inversion row maximizes "gated stitched - ungated stitched" over stale / gamma scenes (scene_pick_v8.json)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap  # noqa: F401,E402

from cdkit import RUNS  # noqa: E402
from cdkit.index import UnifiedPairedDataset  # noqa: E402
from cdkit.ispgate import maybe_invert  # noqa: E402
from cdkit.realdeg import apply_any, menu as build_menu  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from rgbta.models.router import compose  # noqa: E402
from c_eval import DomRetina, load_system  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from det_cam_util import composite, det_cam  # noqa: E402
from precompute_region_labels import draw_rng  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402
import c_sota2  # noqa: E402
import c_v8_data as D  # noqa: E402

OUT = Path(RUNS) / "figures_c"; device = "cuda"; DOM = "lynred_mds"
ACT = np.array([[0.72, 0.72, 0.72], [0.0, 0.447, 0.698], [0.902, 0.624, 0.0]])
SETS = {"A": [("real_agc_stale", "Real AGC: stale window"), ("real_agc_histeq", "Real AGC: histogram eq."), ("ir_stripe_nuc", "IR stripe NUC")],
        "B": [("real_agc_minmax", "Real AGC: min-max"), ("real_agc_gamma", "Real AGC: gamma"), ("ir_dead_pixels", "IR dead pixels")]}
COLS = ["VIS input", "TIR as emitted", "CDDFuse", "Text-IF, matched prompt", "ControlFusion", "Ours"]
INVCOLS = ["VIS input", "TIR as emitted", "TIR after gated inversion", "Joint fusion, as emitted", "Joint fusion, after inversion", "Ours"]


def pick_scenes():
    ev = "dom_rtdetr"; b = D.rows("tau0.04_extra", ev); f = D.rows(D.INV, ev); s2 = D.rows("sota2", ev); s3 = D.rows("sota3", ev)
    D.aligned(b, f); D.aligned(b, s2); D.aligned(b, s3)
    pub = np.max(np.stack([s2["cddfuse"], s2["emma"], s2["textif"], s2["textif_p"], s3["controlfusion"], s3["drmf"], b["tardal"], b["seafusion"]]), axis=0)
    agg = defaultdict(lambda: defaultdict(list))
    for i in range(len(b["kind"])):
        k = (b["kind"][i], b["image_id"][i]); a = agg[k]
        a["full"].append(f["composed"][i]); a["sel"].append(b["composed"][i]); a["pub"].append(pub[i]); a["ir"].append(b["ir"][i]); a["inv"].append(f["inverted"][i])
    old = json.loads((OUT / "scene_pick_v7.json").read_text()); pick, used = {}, set()
    for kind in ("ir_stripe_nuc", "ir_dead_pixels"):
        pick[kind] = old[kind]; used.add(old[kind])
    for kind in ("real_agc_stale", "real_agc_histeq", "real_agc_minmax", "real_agc_gamma"):
        cand = [(np.mean(a["full"]) - np.mean(a["pub"]), iid) for (k, iid), a in agg.items()
                if k == kind and len(a["full"]) >= 3 and np.mean(a["inv"]) > 0.5 and np.mean(a["full"]) >= np.mean(a["ir"]) and iid not in used]
        pick[kind] = max(cand)[1]; used.add(pick[kind])
    cand = [(np.mean(a["full"]) - np.mean(a["sel"]), k, iid) for (k, iid), a in agg.items()
            if k in ("real_agc_stale", "real_agc_gamma") and len(a["full"]) >= 4 and np.mean(a["inv"]) > 0.5 and iid not in used]
    best = max(cand); pick["inversion_row"] = {"kind": best[1], "iid": best[2]}
    (OUT / "scene_pick_v8.json").write_text(json.dumps(pick, indent=1)); return pick


class _Wrap:
    def __init__(self, det):
        self.det = det


def draw_boxes(ax, s, name):
    for b in s["boxes"]:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#E69F00", lw=0.7))
    p = s["dets"][name]; keep = p["scores"].cpu().numpy() >= 0.4
    for b in p["boxes"].cpu().numpy()[keep]:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#009E73", lw=0.9))


def action_inset(ax, act):
    ins = ax.inset_axes([0.015, 0.02, 0.24, 0.24]); ins.imshow(ACT[act], interpolation="nearest"); ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor("white"); sp.set_linewidth(0.8)


def main():
    pick = pick_scenes(); print("pick", pick, flush=True)
    critics, fusion, router, _ = load_system(DOM, 0, device); tau = 0.04
    ev = DomRetina(DOM, device); fm = _Wrap(ev.m); raw = RawAccess(DOM); m = build_menu(with_real=True)
    fus = {k: c_sota2.LOADERS[k](device) for k in ("cddfuse", "textif_p", "controlfusion")}

    def scene(kind, iid, label, inv_row=False):
        ds = UnifiedPairedDataset(DOM, ids=[iid], label_mode="super", keep_empty=False); vis, ir, t = ds[0]
        v = vis.permute(1, 2, 0).numpy(); g = ir[0].numpy(); r16 = raw(iid); found = None
        for draw in range(1, 4):
            rng = draw_rng(f"ceval:{DOM}:{iid}", draw); k, sev = m[int(rng.integers(0, len(m)))]
            if k == kind:
                found = apply_any(k, sev, v, g, rng, r16); break
        assert found is not None, (kind, iid)
        vd, idg = found; idg_inv, fired, _ = maybe_invert(kind, idg, r16, "gate2b", 0.02, "emulator", 0.10)
        with torch.no_grad():
            vt = torch.from_numpy(np.ascontiguousarray(vd.transpose(2, 0, 1)))[None].to(device); vg = rgb_to_gray(vt); hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
            T = lambda a: torch.from_numpy(np.ascontiguousarray(a[None][None])).to(device)
            it_e, it_i = T(idg), T(idg_inv)
            fuse = lambda it_: fusion(vg, it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw))
            fused_i = fuse(it_i); qhat = router(vg, it_i); alt = qhat[:, 1:].max(dim=1, keepdim=True).values
            cell = torch.where((alt - qhat[:, :1]) >= tau, 1 + qhat[:, 1:].argmax(dim=1, keepdim=True), torch.zeros_like(alt, dtype=torch.long))
            logits = torch.zeros_like(qhat).scatter_(1, cell, 10.0); composed, _ = compose(logits, fused_i, vg, it_i, hard=True); act = logits.argmax(dim=1)[0].cpu().numpy()
            G = lambda x: x[0, 0].clamp(0, 1).cpu().numpy()
            imgs = {"VIS input": vd, "TIR as emitted": idg, "Ours": G(composed)}
            if inv_row:
                imgs.update({"TIR after gated inversion": idg_inv, "Joint fusion, as emitted": G(fuse(it_e)), "Joint fusion, after inversion": G(fused_i)})
            else:
                imgs.update({"CDDFuse": G(fus["cddfuse"](vg, it_e)), "Text-IF, matched prompt": G(fus["textif_p"](vt, it_e, kind)), "ControlFusion": G(fus["controlfusion"](vt, it_e, kind))})
            dets, cams, qs = {}, {}, {}
            for name, im in imgs.items():
                g3 = im.astype(np.float32) if im.ndim == 2 else (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]).astype(np.float32)
                x3 = torch.from_numpy(np.stack([g3] * 3))[None].to(device) if im.ndim == 2 else vt
                dets[name] = ev.predict(x3); qs[name] = float(np.mean(ev.box_q(x3, t)))
                if name != "VIS input":
                    cams[name] = det_cam(fm, g3)
        s = dict(label=label, iid=iid, kind=kind, imgs=imgs, dets=dets, cams=cams, qs=qs, act=act, boxes=t["boxes"].numpy(), switch=float((act != 0).mean()), fired=bool(fired),
                 cols=INVCOLS if inv_row else COLS)
        print(label, iid, {k: round(v_, 3) for k, v_ in qs.items()}, f"switch {s['switch']:.2f} inverted {fired}", flush=True)
        return s

    sets = {sid: [scene(k, pick[k], lab) for k, lab in cases] for sid, cases in SETS.items()}
    ir = pick["inversion_row"]; lab = {"real_agc_stale": "Gated inversion, stale window", "real_agc_gamma": "Gated inversion, gamma"}[ir["kind"]]
    sets["A"].append(scene(ir["kind"], ir["iid"], lab, inv_row=True))
    for sid, (fq, fc) in {"A": ("figC7_qualitative_v8", "figC9_cam_v8"), "B": ("figC12_qualitative2_v8", "figC13_cam2_v8")}.items():
        rows = sets[sid]
        fig, axs = plt.subplots(len(rows), 6, figsize=(13.2, 2.12 * len(rows)))
        for a, s in zip(axs, rows):
            for ax, name in zip(a, s["cols"]):
                im = s["imgs"][name]
                ax.imshow(im, cmap="gray", vmin=0, vmax=1) if im.ndim == 2 else ax.imshow(im)
                draw_boxes(ax, s, name)
                if name == "Ours":
                    action_inset(ax, s["act"])
                ttl = f"{name}  q={s['qs'][name]:.2f}" + (f", switch {100 * s['switch']:.0f}%" + (", inverted" if s["fired"] else "") if name == "Ours" else "")
                ax.set_title((s["label"] + "\n" if name == "VIS input" else "") + ttl, fontsize=7.4); ax.set_axis_off()
        fig.subplots_adjust(left=0.004, right=0.996, bottom=0.004, top=1 - 0.30 / (2.12 * len(rows)), wspace=0.03, hspace=0.2)   # inset axes do not work with tight_layout
        fig.savefig(OUT / f"{fq}.pdf"); fig.savefig(OUT / f"{fq}.png", dpi=300); plt.close(fig); print("saved", fq)
        fig, axs = plt.subplots(len(rows), 5, figsize=(11.5, 2.2 * len(rows)))
        for a, s in zip(axs, rows):
            for ax, name in zip(a, s["cols"][1:]):
                im = s["imgs"][name]; ax.imshow(composite(im.astype(np.float32), s["cams"][name])); ax.set_axis_off()
                ax.set_title((s["label"] + "\n" if name == "TIR as emitted" else "") + name, fontsize=7.8)
        fig.subplots_adjust(left=0.004, right=0.996, bottom=0.004, top=1 - 0.30 / (2.2 * len(rows)), wspace=0.03, hspace=0.2)
        fig.savefig(OUT / f"{fc}.pdf"); fig.savefig(OUT / f"{fc}.png", dpi=300); plt.close(fig); print("saved", fc)
    json.dump({sid: [{"iid": s["iid"], "kind": s["kind"], "label": s["label"], "qs": s["qs"], "switch": s["switch"], "inverted": s["fired"]} for s in rows] for sid, rows in sets.items()},
              open(OUT / "qual_selection_c_v8.json", "w"), indent=1)


if __name__ == "__main__":
    main()
