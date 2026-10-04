# -*- coding: utf-8 -*-
"""COCO-mAP protocol (false positives penalised) comparing the ways a candidate is consumed,
answering whether late fusion really wins.
Candidates: vis / ir / fused(joint) / composed (stitched image) / late (VIS+IR detected separately, union + class-wise NMS) /
      decision (fused/vis/ir each detected once; keep detections whose centre falls in the matching routed region, then NMS).
Metrics: mAP@[.5:.95], mAP50 (torchmetrics), per-frame false positives FP/frame (score>=0.5 and IoU<0.5 with every GT), clean and stress separately.
Usage: python c_eval_coco.py --system lynred_mds --seed 0 --domain lynred_mds --tau 0.04 --evaluators dom_retinanet,dom_rtdetr,dom_yolo --draws 3
Output: runs/c_coco/<system>__<domain>_s<seed>[_tag]/report.json"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms, box_iou

import _bootstrap  # noqa: F401
from cdkit import RUNS, SPLITS  # noqa: E402
from cdkit.index import UnifiedPairedDataset, collate_pairs  # noqa: E402
from cdkit.realdeg import apply_any, menu as build_menu  # noqa: E402
from cdkit.ispgate import maybe_invert  # noqa: E402
from rgbta.models.fusion_net import rgb_to_gray  # noqa: E402
from rgbta.models.router import compose  # noqa: E402
from rgbta.utils.common import get_logger  # noqa: E402
from c_eval import DomRetina, UltraEval, load_system  # noqa: E402
from c_precompute_labels import RawAccess  # noqa: E402
from precompute_region_labels import draw_rng  # noqa: E402
from train_fusion_joint import det_attention  # noqa: E402

CANDS = ("vis", "ir", "fused", "composed", "late", "decision")


def nms_merge(preds, iou=0.5):
    boxes = torch.cat([p["boxes"] for p in preds]); scores = torch.cat([p["scores"] for p in preds]); labels = torch.cat([p["labels"] for p in preds])
    if len(boxes):
        keep = batched_nms(boxes, scores, labels, iou); boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    return {"boxes": boxes, "scores": scores, "labels": labels}


def filter_by_action(pred, act_map, action, sx, sy):
    if not len(pred["boxes"]):
        return pred
    cx = ((pred["boxes"][:, 0] + pred["boxes"][:, 2]) / 2 * sx).long().clamp(0, act_map.shape[1] - 1)
    cy = ((pred["boxes"][:, 1] + pred["boxes"][:, 3]) / 2 * sy).long().clamp(0, act_map.shape[0] - 1)
    k = act_map[cy, cx] == action
    return {"boxes": pred["boxes"][k], "scores": pred["scores"][k], "labels": pred["labels"][k]}


def fp_count(pred, gt, thr=0.5, iou=0.5):
    k = pred["scores"] >= thr
    b = pred["boxes"][k]
    if len(b) == 0:
        return 0
    if len(gt["boxes"]) == 0:
        return int(len(b))
    m = box_iou(b, gt["boxes"]).max(dim=1).values
    return int((m < iou).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="lynred_mds"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", default="lynred_mds"); ap.add_argument("--split", default="test")
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--menu", choices=["synthetic", "both"], default="both")
    ap.add_argument("--evaluators", default="dom_retinanet,dom_rtdetr,dom_yolo"); ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--invert", choices=["none", "gate", "gate2", "gate2b", "oracle"], default="none"); ap.add_argument("--gate-thr", type=float, default=0.02); ap.add_argument("--canon", choices=["linear_pct", "emulator"], default="linear_pct"); ap.add_argument("--gate2-thr", type=float, default=0.10)
    ap.add_argument("--extra", default="", help="additional always-fuse candidates (published fusion models etc., same names as c_eval --extra): tardal,seafusion,cddfuse,emma,textif,textif_p,controlfusion,drmf")
    args = ap.parse_args(); device = "cuda"
    from c_eval import load_extra_fusers
    extra_names = [x for x in args.extra.split(",") if x]
    CANDS = globals()["CANDS"] + tuple(extra_names)
    out = Path(RUNS) / "c_coco" / (f"{args.system}__{args.domain}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")); out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"c_coco_{args.domain}_{args.tag}", str(out / "run.log"))
    critics, fusion, router, tau = load_system(args.system, args.seed, device); tau = args.tau
    fusers = load_extra_fusers(extra_names, device, args.system)
    evs = {}
    for name in args.evaluators.split(","):
        if name == "dom_retinanet":
            evs[name] = DomRetina(args.domain, device)
        elif name.startswith("dom_retinanet_"):
            evs[name] = DomRetina(args.domain, device, tag=name[len("dom_retinanet_"):])
        else:
            evs[name] = UltraEval(Path(RUNS) / "heldout" / f"{args.domain}_{name[4:]}" / "weights" / "best.pt", device, name[4:])
    ids = SPLITS / f"{args.domain}_{args.split}.txt"; ids = ids.read_text(encoding="utf-8").split() if ids.exists() else None
    ds = UnifiedPairedDataset(args.domain, split=None if ids else args.split, ids=ids, label_mode="super", keep_empty=False)
    if args.limit:
        ds.recs = ds.recs[: args.limit]; ds.items = ds.items[: args.limit]
    raw = RawAccess(args.domain); has_raw = any(r.get("ir16") for r in ds.recs)
    m = build_menu(with_real=(args.menu == "both" and has_raw))
    loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate_pairs)
    metrics = {(n, c, blk): MeanAveragePrecision(box_format="xyxy") for n in evs for c in CANDS for blk in ("clean", "stress", "stress_real")}
    fps = defaultdict(list)
    log.info(f"images={len(ds)} tau={tau} evaluators={list(evs)}")
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
                idg, _, _ = maybe_invert(kind, idg, raw16, args.invert, args.gate_thr, args.canon, args.gate2_thr)
                vt = torch.from_numpy(np.ascontiguousarray(vd.transpose(2, 0, 1)))[None].to(device)
                it_ = torch.from_numpy(np.ascontiguousarray(idg[None][None])).to(device)
                vg = rgb_to_gray(vt); hw = (vt.shape[-2] // 4, vt.shape[-1] // 4)
                fused = fusion(vg, it_, det_attention(critics[0], vt, hw), det_attention(critics[1], it_.repeat(1, 3, 1, 1), hw))
                qhat = router(vg, it_)
                alt = qhat[:, 1:].max(dim=1, keepdim=True).values
                cell = torch.where((alt - qhat[:, :1]) >= tau, 1 + qhat[:, 1:].argmax(dim=1, keepdim=True), torch.zeros_like(alt, dtype=torch.long))
                logits = torch.zeros_like(qhat).scatter_(1, cell, 10.0)
                composed, _ = compose(logits, fused, vg, it_, hard=True)
                act_map = logits.argmax(dim=1)[0].cpu(); H8, W8 = act_map.shape; sy, sx = H8 / vg.shape[-2], W8 / vg.shape[-1]
                imgs = {"vis": vt, "ir": it_.repeat(1, 3, 1, 1), "fused": fused.clamp(0, 1).repeat(1, 3, 1, 1), "composed": composed.clamp(0, 1).repeat(1, 3, 1, 1)}
                for fn, f in fusers.items():
                    o = f(vt, it_, kind) if getattr(f, "rgb", False) else f(vg, it_)
                    imgs[fn] = o.clamp(0, 1).repeat(1, 3, 1, 1)
                gt = {"boxes": t["boxes"], "labels": t["labels"]}
                blks = ["clean"] if kind == "clean" else (["stress", "stress_real"] if kind.startswith("real") else ["stress"])
                for n, ev in evs.items():
                    p = {c: ev.predict(img) for c, img in imgs.items()}
                    p["late"] = nms_merge([p["vis"], p["ir"]])
                    p["decision"] = nms_merge([filter_by_action(p["fused"], act_map, 0, sx, sy), filter_by_action(p["vis"], act_map, 1, sx, sy),
                                               filter_by_action(p["ir"], act_map, 2, sx, sy)])
                    for c in CANDS:
                        for blk in blks:
                            metrics[(n, c, blk)].update([p[c]], [gt]); fps[(n, c, blk)].append(fp_count(p[c], gt))
            if (bi + 1) % 50 == 0:
                log.info(f"{bi+1}/{len(loader)}")
    rep = {}
    for n in evs:
        rep[n] = {}
        for blk in ("clean", "stress", "stress_real"):
            rep[n][blk] = {}
            for c in CANDS:
                r = metrics[(n, c, blk)].compute()
                rep[n][blk][c] = {"map": round(float(r["map"]), 4), "map50": round(float(r["map_50"]), 4),
                                  "fp_per_frame": round(float(np.mean(fps[(n, c, blk)])), 3) if fps[(n, c, blk)] else None}
            log.info(f"[{n}] {blk}: " + " | ".join(f"{c} mAP {rep[n][blk][c]['map']:.3f} FP {rep[n][blk][c]['fp_per_frame']}" for c in CANDS))
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log.info("done")


if __name__ == "__main__":
    main()
