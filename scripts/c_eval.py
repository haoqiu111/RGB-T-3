# -*- coding: utf-8 -*-
"""Final evaluation / cross-dataset deployment of a selective-fusion system.

--system  which domain's system to load (fusion joint + critics + router v2 + tau): lynred_mds, or flir
          (the FLIR-trained system from the companion project)
--domain  evaluation domain (any indexed domain, super-class protocol)
--evaluators  comma-separated: dom_retinanet (the evaluation domain's own hold-out RetinaNet), dom_rtdetr, dom_yolo,
              flir_retinanet / flir_rtdetr / flir_yolo (FLIR hold-out evaluators applied zero-shot)
--menu  synthetic | real | both (real = real AGC family on the 16-bit counts, 16-bit domains only)
Output: runs/c_eval/<system>__<domain>/<split>_<menu>_s<seed>[_tag]/{<evaluator>_report.json, <evaluator>_regions.csv, summary.json}
Usage: python c_eval.py --system lynred_mds --domain lynred_mds --split test --draws 3 --evaluators dom_retinanet,flir_rtdetr,flir_yolo --menu both"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from cdkit.ispgate import maybe_invert  # noqa: E402
from cdkit.realdeg import REAL_KINDS, apply_any, menu as build_menu  # noqa: E402
from rgbta.eval.utility import box_qualities  # noqa: E402
from rgbta.models.fusion_net import FusionNet, rgb_to_gray  # noqa: E402
from rgbta.models.router import RouterNetV2, compose  # noqa: E402
from rgbta.utils.common import RUNS as A_RUNS, get_logger  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from eval_heldout import RTDETREvaluator, RetinaEvaluator, YOLOEvaluator  # noqa: E402
from gate_a0 import load_critic  # noqa: E402
from precompute_region_labels import draw_rng  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402
from train_heldout_retinanet import build_retinanet  # noqa: E402


class DomRetina:
    def __init__(self, domain, device, tag=""):
        ck = torch.load(Path(RUNS) / "heldout" / (f"{domain}_super_retinanet" + (f"_{tag}" if tag else "")) / "best.pth", map_location="cpu", weights_only=False)
        self.m = build_retinanet(ck["num_classes"]).to(device).eval(); self.m.load_state_dict(ck["model"])

    @torch.no_grad()
    def predict(self, img3):
        p = self.m(list(img3))[0]
        return {"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu() - 1}

    @torch.no_grad()
    def box_q(self, img3, target):
        return box_qualities(self.predict(img3), {"boxes": target["boxes"], "labels": target["labels"]})


class UltraEval:
    def __init__(self, weights, device, arch):
        from ultralytics import RTDETR, YOLO
        self.m = (RTDETR if arch == "rtdetr" else YOLO)(str(weights)); self.device = device

    @torch.no_grad()
    def predict(self, img3):
        arr = (img3[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        b = self.m.predict(arr[:, :, ::-1], verbose=False, device=self.device)[0].boxes
        return {"boxes": torch.as_tensor(b.xyxy.cpu().numpy(), dtype=torch.float32),
                "scores": torch.as_tensor(b.conf.cpu().numpy(), dtype=torch.float32),
                "labels": torch.as_tensor(b.cls.cpu().numpy(), dtype=torch.int64)}

    @torch.no_grad()
    def box_q(self, img3, target):
        return box_qualities(self.predict(img3), {"boxes": target["boxes"], "labels": target["labels"]})


def late_fusion_q(ev, vt, it3, target, iou=0.5):
    """Decision-level late fusion: detect on VIS and IR separately, merge, class-wise NMS, then per-GT-box quality."""
    from torchvision.ops import batched_nms
    pv, pi = ev.predict(vt), ev.predict(it3)
    boxes = torch.cat([pv["boxes"], pi["boxes"]]); scores = torch.cat([pv["scores"], pi["scores"]]); labels = torch.cat([pv["labels"], pi["labels"]])
    if len(boxes):
        keep = batched_nms(boxes, scores, labels, iou); boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    return box_qualities({"boxes": boxes, "scores": scores, "labels": labels}, {"boxes": target["boxes"], "labels": target["labels"]})


def load_extra_fusers(names, device, system):
    """Additional always-fuse systems: avg / max / gp / tardal / seafusion / published models. Returns {name: fuse(vg, ir)->[1,1,H,W]}"""
    out = {}
    for n in names:
        if n == "avg":
            out[n] = lambda vg, ir: (vg + ir) / 2
        elif n == "max":
            out[n] = lambda vg, ir: torch.maximum(vg, ir)
        elif n == "gp":
            base = Path(A_RUNS) if system == "flir" else Path(RUNS)
            net = FusionNet("upstream").to(device).eval()
            net.load_state_dict(torch.load(base / "fusion" / f"{system}_gp" / "best.pth", map_location="cpu", weights_only=False)["model"])
            out[n] = (lambda net: (lambda vg, ir: net(vg, ir)))(net)
        elif n in ("tardal", "seafusion"):
            from eval_sota import load_seafusion, load_tardal
            out[n] = load_tardal(device) if n == "tardal" else load_seafusion(device)
        elif n in ("cddfuse", "emma", "textif", "textif_p", "textif_agc", "controlfusion", "controlfusion_p", "drmf"):
            from c_sota2 import LOADERS
            out[n] = LOADERS[n](device)
    return out


def load_system(system, seed, device):
    if system == "flir":
        base, rdir = Path(A_RUNS), Path(A_RUNS) / "router" / "flir_router_v2"
    else:
        base, rdir = Path(RUNS), Path(RUNS) / "router" / f"{system}_router_v2_s{seed}"
    critics = [load_critic(base / "critics" / f"{system}_vis" / "best.pth", 3, device),
               load_critic(base / "critics" / f"{system}_ir" / "best.pth", 3, device)]
    fusion = FusionNet("upstream").to(device).eval()
    fusion.load_state_dict(torch.load(base / "fusion" / f"{system}_joint" / "best.pth", map_location="cpu", weights_only=False)["model"])
    router = RouterNetV2().to(device).eval()
    router.load_state_dict(torch.load(rdir / "best.pth", map_location="cpu", weights_only=False)["model"])
    tp = rdir / ("tau_seed0.json" if (rdir / "tau_seed0.json").exists() else "tau.json")
    return critics, fusion, router, json.loads(tp.read_text())["tau"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="lynred_mds"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--menu", choices=["synthetic", "real", "both"], default="both")
    ap.add_argument("--evaluators", default="dom_retinanet,flir_rtdetr,flir_yolo")
    ap.add_argument("--subsample", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tau", type=float, default=None); ap.add_argument("--tag", default="")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--fuse-backbone", default="", help="replace the FUSE candidate by one of the external fusion models given in --extra (e.g. controlfusion)")
    ap.add_argument("--extra", default="", help="additional always-fuse candidates: avg,max,gp,tardal,seafusion,late")
    ap.add_argument("--invert", choices=["none", "gate", "gate2", "gate2b", "oracle"], default="none", help="invert, then fuse: re-execute the canonical AGC on real-AGC frames; gate2b = blind two-level gate that never sees the degradation label")
    ap.add_argument("--gate-thr", type=float, default=0.02); ap.add_argument("--canon", choices=["linear_pct", "emulator"], default="linear_pct"); ap.add_argument("--gate2-thr", type=float, default=0.10)
    args = ap.parse_args(); device = "cuda"
    out = Path(RUNS) / "c_eval" / f"{args.system}__{args.domain}" / (f"{args.split}_{args.menu}_s{args.seed}" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"c_eval_{args.system}_{args.domain}_{args.tag}", str(out / "run.log"))
    critics, fusion, router, tau = load_system(args.system, args.seed, device)
    extra_names = [x for x in args.extra.split(",") if x]
    fusers = load_extra_fusers([x for x in extra_names if x not in ("late", "dgate")], device, args.system)
    dgate = None
    if "dgate" in extra_names:
        import joblib
        from c_descriptor_gate import pair_descriptor
        dgate = joblib.load(Path(RUNS) / "router" / f"{args.system}_dgate.joblib")
    if args.tau is not None:
        tau = args.tau
    evs = {}
    for name in args.evaluators.split(","):
        if name == "dom_retinanet":
            evs[name] = DomRetina(args.domain, device)
        elif name.startswith("dom_retinanet_"):
            evs[name] = DomRetina(args.domain, device, tag=name[len("dom_retinanet_"):])
        elif name in ("dom_rtdetr", "dom_yolo"):
            evs[name] = UltraEval(Path(RUNS) / "heldout" / f"{args.domain}_{name[4:]}" / "weights" / "best.pt", device, name[4:])
        elif name == "flir_retinanet":
            evs[name] = RetinaEvaluator("flir", device)
        elif name == "flir_rtdetr":
            evs[name] = RTDETREvaluator(device)
        elif name == "flir_yolo":
            evs[name] = YOLOEvaluator(device)
    ids = (SPLITS / f"{args.domain}_{args.split}.txt")
    ids = ids.read_text(encoding="utf-8").split() if ids.exists() else None
    ds = UnifiedPairedDataset(args.domain, split=None if ids else args.split, ids=ids, label_mode="super", keep_empty=False)
    if args.subsample and args.subsample < len(ds.recs):
        import hashlib
        ds.recs = sorted(ds.recs, key=lambda r: hashlib.sha256(f"{args.domain}:{r['id']}".encode()).hexdigest())[: args.subsample]
        ds.recs = sorted(ds.recs, key=lambda r: r["id"]); ds.items = [r["id"] for r in ds.recs]
    if args.limit:
        ds.recs = ds.recs[: args.limit]; ds.items = ds.items[: args.limit]
    raw = RawAccess(args.domain)
    has_raw = any(r.get("ir16") for r in ds.recs)
    if args.menu == "real" and not has_raw:
        raise SystemExit("this domain has no 16-bit frames; the real menu is unavailable")
    m = [(k, 0) for k in REAL_KINDS] if args.menu == "real" else build_menu(with_real=(args.menu == "both" and has_raw))
    log.info(f"system={args.system} s{args.seed} tau={tau} domain={args.domain} split={args.split} images={len(ds)} menu={len(m)} evaluators={list(evs)}")
    loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_pairs)
    rows = {n: [] for n in evs}
    with torch.no_grad():
        for bi, (vis, ir, targets) in enumerate(loader):
            t = targets[0]
            if len(t["boxes"]) == 0:
                continue
            v_np = vis[0].permute(1, 2, 0).numpy(); i_np = ir[0, 0].numpy(); raw16 = raw(t["image_id"])
            for draw in range(args.draws + 1):
                rng = draw_rng(f"ceval:{args.domain}:{t['image_id']}", draw)
                if draw == 0:
                    kind, sev, vd, idg = "clean", 0, v_np, i_np
                else:
                    kind, sev = m[int(rng.integers(0, len(m)))]
                    if kind.startswith("real_agc") and raw16 is None:
                        kind, sev = m[0]
                    vd, idg = apply_any(kind, sev, v_np, i_np, rng, raw16)
                idg, inverted, gres = maybe_invert(kind, idg, raw16, args.invert, args.gate_thr, args.canon, args.gate2_thr)
                vt = torch.from_numpy(np.ascontiguousarray(vd.transpose(2, 0, 1)))[None].to(device)
                it_ = torch.from_numpy(np.ascontiguousarray(idg[None][None])).to(device)
                vg = rgb_to_gray(vt); hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
                fused = fusion(vg, it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw))
                if args.fuse_backbone:   # replace the FUSE candidate by an external fusion model; router and tau reused zero-shot (is selection orthogonal to the fusion backbone?)
                    fb = fusers[args.fuse_backbone]
                    fused = (fb(vt, it_, kind) if getattr(fb, "rgb", False) else fb(vg, it_)).clamp(0, 1)
                qhat = router(vg, it_)
                alt = qhat[:, 1:].max(dim=1, keepdim=True).values
                cell = torch.where((alt - qhat[:, :1]) >= tau, 1 + qhat[:, 1:].argmax(dim=1, keepdim=True), torch.zeros_like(alt, dtype=torch.long))
                logits = torch.zeros_like(qhat).scatter_(1, cell, 10.0)
                composed, _ = compose(logits, fused, vg, it_, hard=True)
                cands = {"composed": composed.clamp(0, 1).repeat(1, 3, 1, 1), "fused": fused.clamp(0, 1).repeat(1, 3, 1, 1),
                         "vis": vt, "ir": it_.repeat(1, 3, 1, 1)}
                for fn, f in fusers.items():
                    o = f(vt, it_, kind) if getattr(f, "rgb", False) else f(vg, it_)
                    cands[fn] = o.clamp(0, 1).repeat(1, 3, 1, 1)
                H8, W8 = qhat.shape[-2:]; sy, sx = H8 / vg.shape[-2], W8 / vg.shape[-1]; act_map = logits.argmax(dim=1)[0]
                for n, ev in evs.items():
                    qs = {c: ev.box_q(img, t) for c, img in cands.items()}
                    if "late" in extra_names and hasattr(ev, "predict"):
                        qs["late"] = late_fusion_q(ev, vt, it_.repeat(1, 3, 1, 1), t)
                    if dgate is not None:
                        pick = dgate["labels"][int(dgate["clf"].predict(pair_descriptor(vd, idg)[None])[0])]
                        qs["dgate"] = qs[pick]
                    if any(v is None for v in qs.values()):
                        continue
                    for k in range(len(qs["composed"])):
                        b = t["boxes"][k]; x1, y1 = int(b[0] * sx), int(b[1] * sy)
                        x2, y2 = max(int(b[2] * sx), x1 + 1), max(int(b[3] * sy), y1 + 1)
                        patch = act_map[y1:y2, x1:x2].flatten(); a = int(torch.mode(patch).values) if patch.numel() else 0
                        rows[n].append({"image_id": t["image_id"], "kind": kind, "severity": sev, "box_idx": k, "label": int(t["labels"][k]),
                                        "action": a, "inverted": int(inverted), "gate_resid": (round(gres, 4) if gres is not None else ""), "q_composed": float(qs["composed"][k]), "q_decision": float(qs[("fused", "vis", "ir")[a]][k]),
                                        "q_fused": float(qs["fused"][k]), "q_vis": float(qs["vis"][k]), "q_ir": float(qs["ir"][k]),
                                        **{f"q_{x}": float(qs[x][k]) for x in qs if x not in ("composed", "fused", "vis", "ir")}})
            if (bi + 1) % 50 == 0:
                log.info(f"{bi+1}/{len(loader)}")
    summary = {}
    for n, rs in rows.items():
        if not rs:
            continue
        extra_cols = [k[2:] for k in rs[0] if k.startswith("q_") and k[2:] not in ("composed", "decision", "fused", "vis", "ir")]
        arr = {c: np.array([r[f"q_{c}"] for r in rs]) for c in ("composed", "decision", "fused", "vis", "ir") + tuple(extra_cols)}
        best = np.maximum(arr["vis"], arr["ir"]); acts = np.array([r["action"] for r in rs])
        kinds = np.array([r["kind"] for r in rs]); stress = kinds != "clean"; real = np.array([k.startswith("real") for k in kinds])
        inv = np.array([r.get("inverted", 0) for r in rs], dtype=float)

        def block(mask):
            if int(mask.sum()) == 0:
                return {"n": 0}
            o = {c: round(float(arr[c][mask].mean()), 4) for c in arr}
            o.update(best_single=round(float(best[mask].mean()), 4),
                     nfr_alwaysfuse=round(float((arr["fused"][mask] < best[mask] - args.margin).mean()), 4),
                     nfr_decision=round(float((arr["decision"][mask] < best[mask] - args.margin).mean()), 4),
                     oracle_uplift_rel=round(float((best[mask].mean() - arr["fused"][mask].mean()) / max(arr["fused"][mask].mean(), 1e-6)), 4),
                     decision_gain=round(float((arr["decision"][mask] - arr["fused"][mask]).mean()), 4),
                     switch_rate=round(float((acts[mask] != 0).mean()), 4), n=int(mask.sum()),
                     invert_rate=round(float(inv[mask].mean()), 4))
            for c in extra_cols:
                o[f"nfr_{c}"] = round(float((arr[c][mask] < best[mask] - args.margin).mean()), 4)
            return o
        rep = {"clean": block(~stress), "stress": block(stress), "stress_synthetic": block(stress & ~real), "stress_real": block(real),
               "per_kind": {k: block(kinds == k) for k in sorted(set(kinds.tolist()))}}
        (out / f"{n}_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        with open(out / f"{n}_regions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
        summary[n] = {k: rep[k] for k in ("clean", "stress", "stress_synthetic", "stress_real")}
        log.info(f"[{n}] stress={json.dumps(rep['stress'])} real={json.dumps(rep['stress_real'])}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done")


if __name__ == "__main__":
    main()
