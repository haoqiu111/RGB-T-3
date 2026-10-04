# -*- coding: utf-8 -*-
"""Unified index: converts the native annotations of every domain into registry/index/<domain>.json (schema in cdkit/index.py).

Usage: python p1_build_index.py --domains lynred_mds,mfad,...   (empty = every extracted domain)
Each builder only reads the raw files and never modifies the extracted data. Domains whose class order is not documented officially
(mfad / utokyo / fmb) are flagged in CLASS_NOTES and were checked visually before use.
"""
import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401,E402

from cdkit import EXT, INDEX, RGBT_PROJECT  # noqa: E402
from cdkit.index import save_index  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OLD_EXT = Path(RGBT_PROJECT) / "data" / "extracted"
OLD_SPLITS = Path(RGBT_PROJECT) / "registry" / "splits"

# ---- class tables ----
FMB_CLASSES = ["unlabelled", "Road", "Sidewalk", "Building", "Traffic Lamp", "Traffic Sign",
               "Vegetation", "Sky", "Person", "Car", "Truck", "Bus", "Motorcycle",
                "Bicycle", "Pole"]  # official SegMiF order
FMB_BOX_IDS = [8, 9, 10, 11, 12, 13]
MFNET_CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail",
                 "color_cone", "bump"]
MFNET_BOX_IDS = [1, 2, 3]
MFAD_CLASSES = ["car", "bus", "truck", "pedestrian", "ebike_rider", "cyclist"]  # order checked visually
UTOKYO_CLASSES = ["person", "car", "bike", "color_cone", "car_stop", "bump", "hole",
                  "animal"]  # order checked visually (matches the Karasawa 2017 list)
RLIVIT_CLASSES = ["person", "car", "bicycle", "motorcycle", "truck", "bus", "tramway",
                  "escooter"]
FREIBURG_CLASSES = None  # read from class_names.txt
CLASS_NOTES = {
    "fmb": "class order checked visually (Person/Car/Truck/Motorcycle)",
    "mfad": "6-class order checked visually; column 6 is the per-box illumination level 0-4",
    "utokyo_multispectral": "8-class order checked visually; four bands RGB/NIR/MIR/FIR",
}


def rec(id_, split, scene, vis, ir, ref_w, ref_h, boxes, labels, ir16=None, attrs=None,
        meta=None, extra=None, crop=None, ir_channel=None):
    r = {"id": str(id_), "split": split, "scene": str(scene), "vis": vis, "ir": ir,
         "ir16": ir16, "extra": extra or {}, "ref_w": int(ref_w), "ref_h": int(ref_h),
         "boxes": [[float(v) for v in b] for b in boxes], "labels": [int(l) for l in labels],
         "attrs": attrs or [], "meta": meta or {}}
    if crop is not None:
        r["crop"] = [int(v) for v in crop]
    if ir_channel is not None:
        r["ir_channel"] = int(ir_channel)
    return r


def img_size(p: Path):
    with Image.open(p) as im:
        return im.size


def xywh2xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def yolo_txt(p: Path, w, h):
    boxes, labels, extra = [], [], []
    if not p.exists():
        return boxes, labels, extra
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0])); cx, cy, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        if x2 > x1 and y2 > y1:
            boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)]); labels.append(c)
            extra.append(parts[5:])
    return boxes, labels, extra


def voc_xml(p: Path, class_list):
    boxes, labels = [], []
    root = ET.parse(p).getroot()
    for o in root.iter("object"):
        name = o.find("name").text.strip()
        if name not in class_list:
            continue
        bb = o.find("bndbox")
        x1, y1 = float(bb.find("xmin").text), float(bb.find("ymin").text)
        x2, y2 = float(bb.find("xmax").text), float(bb.find("ymax").text)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2]); labels.append(class_list.index(name))
    return boxes, labels


def mask_boxes(mask: np.ndarray, class_ids, min_area=30):
    """Per-class connected components -> boxes. Touching instances merge into one box."""
    from scipy import ndimage
    boxes, labels = [], []
    for c in class_ids:
        m = mask == c
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        for sl_i, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None:
                continue
            area = int((lab[sl] == sl_i).sum())
            if area < min_area:
                continue
            y1, y2 = sl[0].start, sl[0].stop; x1, x2 = sl[1].start, sl[1].stop
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                boxes.append([x1, y1, x2, y2]); labels.append(c)
    return boxes, labels


# ---------------- builders ----------------
def build_lynred_mds():
    root = EXT / "lynred_mds" / "detection_dataset"
    classes, images = None, []
    for split in ("train", "test"):
        d = json.loads((root / "metadata" / f"ir_{split}.json").read_text(encoding="utf-8"))
        cats = sorted(d["categories"], key=lambda c: c["id"])
        classes = [c["name"] for c in cats]
        cid2idx = {c["id"]: i for i, c in enumerate(cats)}
        anns = defaultdict(list)
        for a in d["annotations_aligned"]:
            anns[a["image_id"]].append(a)
        seq_min = {}
        for im in d["images"]:
            seq_min[im["sequence_id"]] = min(seq_min.get(im["sequence_id"], 10**9), im["id"])
        for im in d["images"]:
            blk = (im["id"] - seq_min[im["sequence_id"]]) // 250  # blocks of 250 frames within a sequence = scene groups
            ir = f"infrared_aligned/{im['file_name']}"
            ir16 = f"infrared_16bits_aligned/{im['file_name']}"
            vis = f"visible_aligned/{im['visible_image']}"
            w, h = img_size(root / ir)
            vw, vh = img_size(root / vis)
            assert abs(vw / w - vh / h) < 0.02, (im["file_name"], w, h, vw, vh)
            boxes, labels, attrs = [], [], []
            for a in anns[im["id"]]:
                b = xywh2xyxy(a["bbox"])
                if b[2] > b[0] and b[3] > b[1]:
                    boxes.append(b); labels.append(cid2idx[a["category_id"]])
                    attrs.append({k: a["attributes"].get(k) for k in ("occlusion", "rider", "special")})
            images.append(rec(Path(im["file_name"]).stem, split, f"seq_{im['sequence_id']}_{blk}",
                              "detection_dataset/" + vis, "detection_dataset/" + ir, w, h,
                              boxes, labels, ir16="detection_dataset/" + ir16, attrs=attrs,
                              meta={"tod": im["time_of_day"], "season": im["season"],
                                    "tamb": im["tamb"], "author": im["author"],
                                    "ir16_unaligned": f"detection_dataset/infrared_16bits/{im['file_name']}",
                                    "vis_w": vw, "vis_h": vh}))
    return {"domain": "lynred_mds", "classes": classes,
            "note": "boxes in the aligned IR frame (annotations_aligned); vis_aligned is twice that size", "images": images}


def build_mfad():
    root = EXT / "mfad" / "MFAD"
    images = []
    for split in ("train", "test"):
        for lp in sorted((root / "labels" / split).glob("*.txt")):
            stem = lp.stem
            vis = f"MFAD/visible/{split}/{stem}.jpg"; ir = f"MFAD/infrared/{split}/{stem}.jpg"
            if not (EXT / "mfad" / ir).exists():
                continue
            w, h = img_size(EXT / "mfad" / ir)
            boxes, labels, ex = yolo_txt(lp, w, h)
            attrs = [{"illum": int(e[0]) if e else None} for e in ex]
            scene = stem[7:15] if stem.startswith("cali_l_") else stem[:8]
            images.append(rec(stem, split, scene, vis, ir, w, h, boxes, labels, attrs=attrs,
                              meta={"illum_img": max((a["illum"] for a in attrs if a["illum"] is not None), default=None)}))
    return {"domain": "mfad", "classes": MFAD_CLASSES, "note": CLASS_NOTES["mfad"], "images": images}


def build_smod():
    root = EXT / "smod"
    images, classes = [], None
    for split in ("train", "test"):
        d = json.loads((root / "anno" / f"new_{split}_annotations_rgb.json").read_text(encoding="utf-8"))
        cats = sorted(d["categories"], key=lambda c: c["id"]); classes = [c["name"] for c in cats]
        cid2idx = {c["id"]: i for i, c in enumerate(cats)}
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            vis = im["file_name"]; ir = vis.replace("_rgb.jpg", "_tir.jpg")
            if not (root / ir).exists():
                continue
            boxes, labels, attrs = [], [], []
            for a in anns[im["id"]]:
                b = xywh2xyxy(a["bbox"])
                if b[2] > b[0] and b[3] > b[1]:
                    boxes.append(b); labels.append(cid2idx[a["category_id"]])
                    attrs.append({"occlusion": a.get("occlusion")})
            tod = vis.split("/")[0]; num = int(Path(vis).stem.split("_")[0])
            images.append(rec(f"{tod}_{num:06d}", split, f"{tod}_{num // 250}", vis, ir,
                              im["width"], im["height"], boxes, labels, attrs=attrs, meta={"tod": tod}))
    return {"domain": "smod", "classes": classes, "note": "boxes annotated on the RGB frame; the dataset states strict registration", "images": images}


def build_fmb():
    root = EXT / "fmb"; images = []
    for split in ("train", "test"):
        for lp in sorted((root / split / "Label").glob("*.png")):
            stem = lp.stem
            mask = np.asarray(Image.open(lp))
            if mask.ndim == 3:
                mask = mask[..., 0]
            h, w = mask.shape
            boxes, labels = mask_boxes(mask, FMB_BOX_IDS)
            images.append(rec(f"{split}_{stem}", split, f"{split}_{int(stem) // 50}",
                              f"{split}/Visible/{stem}.png", f"{split}/Infrared/{stem}.png",
                              w, h, boxes, labels, meta={"from_mask": True}))
    return {"domain": "fmb", "classes": FMB_CLASSES, "note": CLASS_NOTES["fmb"] + "; boxes from mask connected components", "images": images}


def build_msrs():
    root = EXT / "msrs" / "MSRS-main"; images = []
    for split in ("train", "test"):
        for lp in sorted((root / split / "Segmentation_labels").glob("*.png")):
            stem = lp.stem
            mask = np.asarray(Image.open(lp))
            if mask.ndim == 3:
                mask = mask[..., 0]
            h, w = mask.shape
            boxes, labels = mask_boxes(mask, MFNET_BOX_IDS)
            tod = "night" if stem.endswith("N") else "day"
            images.append(rec(stem, split, f"{tod}_{int(stem[:-1]) // 50}",
                              f"MSRS-main/{split}/vi/{stem}.png", f"MSRS-main/{split}/ir/{stem}.png",
                              w, h, boxes, labels, meta={"tod": tod, "from_mask": True}))
    return {"domain": "msrs", "classes": MFNET_CLASSES, "note": "enhanced MFNet; boxes from mask connected components", "images": images}


def build_mfnet():
    root = EXT / "mfnet" / "ir_seg_dataset"; images = []
    split_of = {}
    for tag, f in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        p = root / f
        if p.exists():
            for s in p.read_text().split():
                split_of[s] = tag
    for lp in sorted((root / "labels").glob("*.png")):
        stem = lp.stem
        mask = np.asarray(Image.open(lp))
        if mask.ndim == 3:
            mask = mask[..., 0]
        h, w = mask.shape
        boxes, labels = mask_boxes(mask, MFNET_BOX_IDS)
        tod = "night" if stem.endswith("N") else "day"
        img = f"ir_seg_dataset/images/{stem}.png"
        images.append(rec(stem, split_of.get(stem, "train"), f"{tod}_{int(stem[:-1]) // 50}",
                          img, img, w, h, boxes, labels, ir_channel=3,
                          meta={"tod": tod, "from_mask": True, "msrs_twin": stem}))
    return {"domain": "mfnet", "classes": MFNET_CLASSES,
            "note": "4-channel PNG: RGB = first three channels, thermal = channel 4 (ir_channel=3); raw low-SNR infrared", "images": images}


def build_dronevehicle():
    root = EXT / "dronevehicle" / "DroneVehicle"; images = []; names = []
    B = 100  # white border
    for split in ("train", "val", "test"):
        ann_dir = root / split / "annfiles_all"
        for ap in sorted(ann_dir.glob("*.txt")):
            stem = ap.stem
            vis = f"DroneVehicle/{split}/{split}img/{stem}.jpg"; ir = f"DroneVehicle/{split}/{split}imgr/{stem}.jpg"
            if not (EXT / "dronevehicle" / ir).exists():
                continue
            boxes, labels = [], []
            for line in ap.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
                parts = line.split()
                if len(parts) < 9:
                    continue
                pts = np.array(list(map(float, parts[:8]))).reshape(4, 2) - B
                name = parts[8].lower().replace(" ", "_")
                if name in ("feright_car", "freight_car", "feright"):
                    name = "freight_car"
                if name not in names:
                    names.append(name)
                x1, y1 = pts.min(0); x2, y2 = pts.max(0)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(640, x2), min(512, y2)
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    boxes.append([x1, y1, x2, y2]); labels.append(names.index(name))
            images.append(rec(f"{split}_{stem}", split, f"{split}_{int(stem) // 100}", vis, ir,
                              640, 512, boxes, labels, crop=[B, B, B + 640, B + 512]))
    return {"domain": "dronevehicle", "classes": names,
            "note": "horizontal box enclosing each OBB; 100 px white border cropped; annfiles_all annotate the infrared frame", "images": images}


def _coco_pairs(domain, json_rel, split, img_dir_vis, img_dir_ir, ref_from_json=True):
    root = EXT / domain
    d = json.loads((root / json_rel).read_text(encoding="utf-8"))
    cats = sorted(d["categories"], key=lambda c: c["id"]); classes = [c["name"] for c in cats]
    cid2idx = {c["id"]: i for i, c in enumerate(cats)}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    out = []
    for im in d["images"]:
        f = im["file_name"]
        ir = f"{img_dir_ir}/{f}"; vis = f"{img_dir_vis}/{f}"
        if not (root / ir).exists():
            continue
        boxes, labels, attrs = [], [], []
        for a in anns[im["id"]]:
            b = xywh2xyxy(a["bbox"])
            if b[2] > b[0] and b[3] > b[1]:
                boxes.append(b); labels.append(cid2idx[a["category_id"]])
                attrs.append({"ignore": a.get("ignore", 0)})
        num = int(Path(f).stem)
        out.append(rec(f"{split}_{Path(f).stem}", split, f"{split}_{num // 100}", vis, ir,
                       im["width"], im["height"], boxes, labels, attrs=attrs))
    return classes, out


def build_rgbt_droneperson():
    c1, a = _coco_pairs("rgbt_droneperson", "train_thermal.json", "train", "train/visible", "train/thermal")
    c2, b = _coco_pairs("rgbt_droneperson", "val_thermal.json", "val", "val/visible", "val/thermal")
    return {"domain": "rgbt_droneperson", "classes": c1, "note": "boxes on the thermal frame; the visible frame is noticeably offset (dataset property)", "images": a + b}


def build_vtuav_det():
    c1, a = _coco_pairs("vtuav_det", "VTUAV_v1.0/train_ir.json", "train", "VTUAV_v1.0/train/rgb", "VTUAV_v1.0/train/ir")
    c2, b = _coco_pairs("vtuav_det", "VTUAV_v1.0/val_ir.json", "test", "VTUAV_v1.0/test/rgb", "VTUAV_v1.0/test/ir")
    return {"domain": "vtuav_det", "classes": c1, "note": "boxes on the thermal frame, 1920x1080", "images": a + b}


def build_rlivit_rgbt():
    root = EXT / "rlivit_rgbt" / "R-LiViT_RGB-T"
    meta = {}
    for s in ET.parse(root / "sequences.xml").getroot().iter("sequence"):
        meta[s.find("rgbt_seq_id").text] = {"tod": s.find("daytime").text, "location": s.find("location").text}
    split_of = {}
    for tag in ("train", "test"):
        for s in (root / f"{tag}.txt").read_text().split():
            split_of[s.strip()] = tag
    images = []
    for seq_dir in sorted((root / "annotations").iterdir()):
        seq = seq_dir.name
        for xp in sorted(seq_dir.glob("*.xml")):
            fr = xp.stem
            ir = f"R-LiViT_RGB-T/thermal/{seq}/{fr}.png"; vis = f"R-LiViT_RGB-T/rgb/{seq}/{fr}.png"
            if not (EXT / "rlivit_rgbt" / ir).exists():
                continue
            boxes, labels = voc_xml(xp, RLIVIT_CLASSES)
            images.append(rec(f"{seq}_{fr}", split_of.get(seq, "train"), f"seq_{seq}", vis, ir,
                              1280, 720, boxes, labels,
                              meta={**meta.get(seq, {}), "seq": seq, "frame": int(fr)}))
    return {"domain": "rlivit_rgbt", "classes": RLIVIT_CLASSES, "note": "fixed roadside camera; 50 frames per sequence, every 4th frame annotated", "images": images}


def build_utokyo():
    root = EXT / "utokyo_multispectral" / "ir_det_dataset"; images = []
    for fp in sorted((root / "Images" / "fir").glob("*.png")):
        stem = fp.stem
        w, h = img_size(fp)
        boxes, labels, _ = yolo_txt(root / "labels" / "fir" / f"{stem}.txt", w, h)
        images.append(rec(stem, "all", f"seq_{int(stem) // 250}",
                          f"ir_seg_dataset_placeholder", f"ir_det_dataset/Images/fir/{stem}.png",
                          w, h, boxes, labels,
                          extra={"nir": f"ir_det_dataset/Images/nir/{stem}.png",
                                 "mir": f"ir_det_dataset/Images/mir/{stem}.png",
                                 "rgb_labels": f"ir_det_dataset/labels/rgb/{stem}.txt"}))
        images[-1]["vis"] = f"ir_det_dataset/Images/rgb/{stem}.png"
    return {"domain": "utokyo_multispectral", "classes": UTOKYO_CLASSES,
            "note": CLASS_NOTES["utokyo_multispectral"] + "; boxes from the FIR-frame annotation", "images": images}


def build_lynred_stereo():
    root = EXT / "lynred_stereo" / "stereo_dataset"; images = []
    meta_txt = (root / "metadata" / "metadata.json").read_text(encoding="utf-8")
    import re
    seq_meta = {}
    for m in re.finditer(r'"sequence_name":\s*"(sequence_\d+)".*?"environment":\s*"(\w+)".*?"time_of_day":\s*"(\w+)"', meta_txt, re.S):
        seq_meta[m.group(1)] = {"env": m.group(2), "tod": m.group(3)}
    for seq_dir in sorted(p for p in root.iterdir() if p.name.startswith("sequence_")):
        seq = seq_dir.name
        for ip in sorted((seq_dir / "sequence_ir_1" / "8bits").glob("*.png")):
            fr = ip.stem
            ir16 = f"stereo_dataset/{seq}/sequence_ir_1/16bits/{fr}.png"
            vis = f"stereo_dataset/{seq}/sequence_vis_1/{fr}.jpg"
            w, h = img_size(ip)
            images.append(rec(f"{seq}_{fr}", "stream", seq, vis if (EXT / "lynred_stereo" / vis).exists() else None,
                              f"stereo_dataset/{seq}/sequence_ir_1/8bits/{fr}.png", w, h, [], [],
                              ir16=ir16 if (EXT / "lynred_stereo" / ir16).exists() else None,
                              meta={**seq_meta.get(seq, {}), "frame": int(fr.split("_")[-1])}))
    return {"domain": "lynred_stereo", "classes": [], "note": "unannotated stream; left thermal + left visible; not pixel-aligned (127 mm stereo baseline)", "images": images}


def build_freiburg():
    root = EXT / "freiburg_thermal"; images = []
    names = (root / "test" / "day" / "class_names.txt").read_text().split("\n")
    names = [n.strip() for n in names if n.strip()]
    box_ids = [i for i, n in enumerate(names) if any(k in n.lower() for k in ("person", "car", "bicycle"))]
    for tod in ("day", "night"):
        for mp in sorted((root / "test" / tod / "SegmentationClass").glob("*.npy")):
            stem = mp.stem  # fl_ir_aligned_<t1>_<t2>_rgb ; the IR file name drops a trailing 0 of t2
            mask = np.load(mp)
            if mask.ndim == 3:
                mask = mask[..., 0]
            h, w = mask.shape
            boxes, labels = mask_boxes(mask, box_ids)
            parts = stem.split("_")
            t1, t2 = parts[3], parts[4]
            ir_cands = [p for p in (root / "test" / tod / "ImagesIR").glob(f"fl_ir_aligned_{t1}_*_ir.png")
                        if t2.startswith(p.stem.split("_")[4])]
            rgb_cands = list((root / "test" / tod / "ImagesRGB").glob(f"{stem}.*"))
            if not ir_cands:
                continue
            irp = ir_cands[0]
            is16 = Image.open(irp).mode.startswith("I")
            X0, X1 = 300, 1710  # valid infrared field of view (registration borders), identical on the 6 training sequences and test
            cb, cl = [], []
            for b, l in zip(boxes, labels):
                x1, x2 = max(0, b[0] - X0), min(X1 - X0, b[2] - X0)
                if x2 - x1 >= 3:
                    cb.append([x1, b[1], x2, b[3]]); cl.append(l)
            images.append(rec(f"test_{tod}_{stem}", "test", f"test_{tod}",
                              rgb_cands[0].relative_to(root).as_posix() if rgb_cands else None,
                              irp.relative_to(root).as_posix(), X1 - X0, h, cb, cl,
                              ir16=irp.relative_to(root).as_posix() if is16 else None,
                              meta={"tod": tod, "from_mask": True}, crop=[X0, 0, X1, h]))
    for seq_dir in sorted((root / "train").iterdir()):
        if not seq_dir.is_dir():
            continue
        tod = "night" if "night" in seq_dir.name else "day"
        for sub in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
            ir_dir = sub / "fl_ir_aligned"; rgb_dir = sub / "fl_rgb"
            rgb_map = {p.stem.replace("fl_rgb_", ""): p for p in rgb_dir.glob("*.png")} if rgb_dir.exists() else {}
            for ip in sorted(ir_dir.glob("*.png")):
                key = ip.stem.replace("fl_ir_aligned_", "")
                rp = rgb_map.get(key)
                images.append(rec(f"{seq_dir.name}_{sub.name}_{key}", "stream", f"{seq_dir.name}/{sub.name}",
                                  rp.relative_to(root).as_posix() if rp else None,
                                  ip.relative_to(root).as_posix(), 1410, 650, [], [],
                                  ir16=ip.relative_to(root).as_posix(), crop=[300, 0, 1710, 650],
                                  meta={"tod": tod, "ts": float(key.replace("_", ".")) if key.replace("_", "").isdigit() else None}))
    return {"domain": "freiburg_thermal", "classes": names,
            "note": "train is an unannotated 16-bit stream; the 66 test frames have boxes from segmentation masks", "images": images}


def build_rgbt234():
    """Tracking benchmark: one target box per frame (infrared.txt: x,y,w,h on the infrared frame); no classes, classes=['target']."""
    root = EXT / "rgbt234" / "RGB_T234"; images = []
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        gt_p = seq_dir / "infrared.txt"
        if not gt_p.exists() or not (seq_dir / "infrared").exists():
            continue
        gts = [l.replace("\t", ",").split(",") for l in gt_p.read_text().strip().splitlines()]
        ir_files = sorted((seq_dir / "infrared").glob("*.jpg"))
        vis_files = sorted((seq_dir / "visible").glob("*.jpg"))
        if not ir_files:
            continue
        w, h = img_size(ir_files[0])
        for k, ip in enumerate(ir_files):
            if k >= len(gts):
                break
            try:
                x, y, bw, bh = (float(v) for v in gts[k][:4])
            except ValueError:
                continue
            boxes = [[x, y, x + bw, y + bh]] if bw > 1 and bh > 1 else []
            vp = vis_files[k] if k < len(vis_files) else None
            images.append(rec(f"{seq_dir.name}_{k:05d}", "stream", seq_dir.name,
                              vp.relative_to(EXT / "rgbt234").as_posix() if vp else None,
                              ip.relative_to(EXT / "rgbt234").as_posix(), w, h, boxes,
                              [0] * len(boxes), meta={"seq": seq_dir.name, "frame": k}))
    return {"domain": "rgbt234", "classes": ["target"],
            "note": "tracking benchmark, one target box per frame (infrared coordinates); per-object utility analysis only", "images": images}


# ---- domains prepared in the companion project (FLIR-aligned, LLVIP, M3FD, KAIST) ----
def build_flir_align():
    root = OLD_EXT / "flir_align_3class" / "FLIR-align-3class"; images = []
    split_ids = {}
    for tag in ("train", "val", "calib"):
        p = OLD_SPLITS / f"flir_{tag}.txt"
        if p.exists():
            for s in p.read_text().split():
                split_ids[s] = tag
    for split in ("train", "test"):
        for vp in sorted((root / "visible" / split).glob("*.jpg")):
            stem = vp.stem
            ir = root / "infrared" / split / f"{stem}.jpeg"
            if not ir.exists():
                continue
            w, h = img_size(vp)
            boxes, labels, _ = yolo_txt(root / "labels" / split / f"{stem}.txt", w, h)
            tod = "night" if stem.endswith("_night") else "day"
            num = int(stem.split("_")[1])
            images.append(rec(stem, split_ids.get(stem, split), f"{tod}_{num // 250}",
                              str(vp), str(ir), w, h, boxes, labels, meta={"tod": tod}))
    return {"domain": "flir_align", "root": str(OLD_EXT / "flir_align_3class" / "FLIR-align-3class"),
            "classes": ["person", "car", "bicycle"], "note": "FLIR-aligned, 3 classes; absolute paths", "images": images}


def build_llvip():
    root = OLD_EXT / "llvip_aligned" / "LLVIP"; images = []
    split_ids = {}
    for tag in ("train", "val", "calib"):
        p = OLD_SPLITS / f"llvip_{tag}.txt"
        if p.exists():
            for s in p.read_text().split():
                split_ids[s] = tag
    for split in ("train", "test"):
        for vp in sorted((root / "visible" / split).glob("*.jpg")):
            stem = vp.stem
            ir = root / "infrared" / split / f"{stem}.jpg"
            if not ir.exists():
                continue
            w, h = img_size(vp)
            boxes, labels = voc_xml(root / "Annotations" / f"{stem}.xml", ["person"])
            images.append(rec(stem, split_ids.get(stem, split), stem[:2], str(vp), str(ir), w, h,
                              boxes, labels, meta={"tod": "night"}))
    return {"domain": "llvip", "root": str(root), "classes": ["person"], "note": "LLVIP", "images": images}


def build_m3fd():
    root = OLD_EXT / "m3fd_detection"; images = []
    cls = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
    split_ids = {}
    for tag in ("train", "eval"):
        p = OLD_SPLITS / f"m3fd_{tag}.txt"
        if p.exists():
            for s in p.read_text().split():
                split_ids[s] = tag
    for vp in sorted((root / "vi").glob("*.png")):
        stem = vp.stem
        w, h = img_size(vp)
        boxes, labels = voc_xml(root / "Annotation" / f"{stem}.xml", cls)
        images.append(rec(stem, split_ids.get(stem, "all"), f"seq_{int(stem) // 250}", str(vp),
                          str(root / "ir" / f"{stem}.png"), w, h, boxes, labels))
    return {"domain": "m3fd", "root": str(root), "classes": ["people", "car", "bus", "motorcycle", "lamp", "truck"],
            "note": "M3FD detection", "images": images}


def build_kaist():
    root = OLD_EXT / "kaist_reannotated"; images = []
    for split in ("train", "test"):
        base = root / f"kaist_wash_picture_{split}"
        for vp in sorted((base / "visible").glob("*.jpg")):
            stem = vp.stem
            w, h = img_size(vp)
            xml = root / f"kaist_wash_annotation_{split}" / f"{stem}.xml"
            boxes, labels = voc_xml(xml, ["person"]) if xml.exists() else ([], [])
            parts = stem.split("_")
            images.append(rec(stem, split, f"{parts[0]}_{parts[1]}", str(vp),
                              str(base / "lwir" / f"{stem}.jpg"), w, h, boxes, labels))
    return {"domain": "kaist", "root": str(root), "classes": ["person"], "note": "KAIST, re-annotated", "images": images}


BUILDERS = {
    "lynred_mds": build_lynred_mds, "mfad": build_mfad, "smod": build_smod, "fmb": build_fmb,
    "msrs": build_msrs, "mfnet": build_mfnet, "dronevehicle": build_dronevehicle,
    "rgbt_droneperson": build_rgbt_droneperson, "vtuav_det": build_vtuav_det,
    "rlivit_rgbt": build_rlivit_rgbt, "utokyo_multispectral": build_utokyo,
    "lynred_stereo": build_lynred_stereo, "freiburg_thermal": build_freiburg, "rgbt234": build_rgbt234,
    "flir_align": build_flir_align, "llvip": build_llvip, "m3fd": build_m3fd, "kaist": build_kaist,
}
OLD_DOMAINS = {"flir_align", "llvip", "m3fd", "kaist"}


def ready(domain):
    if domain in OLD_DOMAINS:
        return True
    return (EXT / domain / ".extract_done").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    doms = [d for d in args.domains.split(",") if d] or list(BUILDERS)
    from cdkit.index import summarize
    for d in doms:
        if not ready(d):
            print(f"[skip] {d}: not extracted"); continue
        if (INDEX / f"{d}.json").exists() and not args.force:
            print(f"[skip] {d}: index exists (use --force to overwrite)"); continue
        t0 = time.time()
        try:
            idx = BUILDERS[d]()
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {d}: {e!r}"); continue
        save_index(d, idx)
        s = summarize(d)
        print(f"[ok] {d}: {s['n_images']} imgs, splits={s['per_split']}, scenes={s['n_scenes']}, "
              f"ir16={s['n_with_ir16']}, per_class={s['per_class']} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
