# -*- coding: utf-8 -*-
"""Unified cross-dataset index (registry/index/<domain>.json) and the common paired loader.

Index schema (one JSON per domain):
{
  "domain": str, "classes": [native class name, ...],   # a native label is an index into this list
  "note": str,
  "root": str | absent,                                  # optional absolute data root; default data/extracted/<domain>
  "images": [
    {"id": str, "split": "train|val|test|stream", "scene": str,
     "vis": relpath|null, "ir": relpath, "ir16": relpath|null,
     "extra": {"nir": relpath, "mir": relpath} | {},        # extra bands etc.
     "ref_w": int, "ref_h": int,                           # coordinate frame of the boxes
     "boxes": [[x1,y1,x2,y2], ...], "labels": [int, ...],
     "attrs": [{...}, ...] | [],                            # per-box attributes (occlusion / illumination)
     "meta": {"tod": "day|night", "season": str, "tamb": float, ...},
     "crop": [x0,y0,x1,y1] | absent,                        # optional crop applied to vis/ir/ir16
     "ir_channel": int | absent}                            # multi-channel PNG: index of the thermal channel
  ]
}
relpath is relative to data/extracted/<domain>/ (or "root").
The loader resizes vis/ir to size=(W,H) and scales boxes by (W/ref_w, H/ref_h);
the output protocol matches rgbta.data.paired_dataset.PairedDetDataset.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cdkit import EXT, INDEX, SUPER_IDS, SUPER_SYNONYMS
from cdkit.thermal16 import load_ir16, tone_map


def load_index(domain: str) -> dict:
    return json.loads((INDEX / f"{domain}.json").read_text(encoding="utf-8"))


def save_index(domain: str, idx: dict) -> Path:
    INDEX.mkdir(parents=True, exist_ok=True)
    p = INDEX / f"{domain}.json"
    p.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    return p


def super_label_map(classes: list[str]) -> dict[int, int]:
    """Native label id -> super-class id (0/1/2); classes outside the table are absent from the map."""
    m = {}
    for i, name in enumerate(classes):
        s = SUPER_SYNONYMS.get(name.strip().lower())
        if s is not None:
            m[i] = SUPER_IDS[s]
    return m


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_gray8(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I", "I;16L"):
        return tone_map(np.asarray(im, dtype=np.float32), "linear_pct:1")
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _resize(a: np.ndarray, size) -> np.ndarray:
    W, H = size
    if (a.shape[1], a.shape[0]) == (W, H):
        return a
    im = Image.fromarray((a * 255).astype(np.uint8))
    return np.asarray(im.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0


class UnifiedPairedDataset(Dataset):
    """domain: a name under registry/index; split: None = all, or 'train'|'val'|'test'|'stream'.

    label_mode: 'native' keeps native class ids; 'super' maps to {0 person, 1 car, 2 bicycle} and drops the rest.
    ir_source: '8bit' uses the released 8-bit infrared; 'raw:<method>' tone-maps the 16-bit frame with tone_map(method).
    vis_fallback: when no visible image exists (thermal-only streams) replicate the infrared into three channels.
    """

    def __init__(self, domain: str, split=None, ids=None, size=(640, 512),
                 label_mode="native", degradation=None, ir_source="8bit",
                 vis_fallback=True, keep_empty=True):
        self.domain, self.size = domain, size
        self.label_mode, self.degradation = label_mode, degradation
        self.ir_source, self.vis_fallback = ir_source, vis_fallback
        idx = load_index(domain)
        self.root = Path(idx["root"]) if idx.get("root") else EXT / domain
        self.classes = idx["classes"]
        self.smap = super_label_map(self.classes)
        recs = idx["images"]
        if split is not None:
            splits = {split} if isinstance(split, str) else set(split)
            recs = [r for r in recs if r["split"] in splits]
        if ids is not None:
            keep = set(ids)
            recs = [r for r in recs if r["id"] in keep]
        if not keep_empty:
            recs = [r for r in recs if len(self._filtered(r)[0])]
        self.recs = sorted(recs, key=lambda r: r["id"])
        self.items = [r["id"] for r in self.recs]

    def __len__(self):
        return len(self.recs)

    def scene_id(self, idx: int) -> str:
        return self.recs[idx]["scene"]

    def meta(self, idx: int) -> dict:
        return self.recs[idx].get("meta", {})

    def _filtered(self, r):
        boxes = np.array(r["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.array(r["labels"], dtype=np.int64).reshape(-1)
        if self.label_mode == "super":
            keep = np.array([l in self.smap for l in labels], dtype=bool)
            boxes, labels = boxes[keep], np.array([self.smap[l] for l in labels[keep]],
                                                  dtype=np.int64)
        return boxes, labels

    def __getitem__(self, idx: int):
        r = self.recs[idx]
        W, H = self.size
        if r.get("ir_channel") is not None:  # multi-channel PNG (MFNet: RGB + thermal as four channels)
            arr = np.asarray(Image.open(self.root / r["ir"]))
            ir = arr[..., r["ir_channel"]].astype(np.float32) / 255.0
            vis = arr[..., :3].astype(np.float32) / 255.0
        else:
            if self.ir_source == "8bit" or not r.get("ir16"):
                ir = _load_gray8(self.root / r["ir"])
            else:
                method = self.ir_source.split(":", 1)[1]
                ir = tone_map(load_ir16(self.root / r["ir16"]), method)
            if r.get("vis"):
                vis = _load_rgb(self.root / r["vis"])
            elif self.vis_fallback:
                vis = np.repeat(ir[..., None], 3, axis=2)
            else:
                raise FileNotFoundError(f"{self.domain}:{r['id']} has no visible image")
        if r.get("crop"):
            x0, y0, x1, y1 = r["crop"]
            vis, ir = vis[y0:y1, x0:x1], ir[y0:y1, x0:x1]
        boxes, labels = self._filtered(r)
        rw, rh = r["ref_w"], r["ref_h"]
        vis, ir = _resize(vis, self.size), _resize(ir, self.size)
        if len(boxes):
            boxes = boxes * np.array([W / rw, H / rh, W / rw, H / rh], dtype=np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
            keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
            boxes, labels = boxes[keep], labels[keep]
        if self.degradation is not None:
            import hashlib
            seed = int(hashlib.sha256(f"{self.domain}:{r['id']}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vis, ir = self.degradation(vis, ir, rng)
        return (torch.from_numpy(np.ascontiguousarray(vis.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(ir[None].astype(np.float32))),
                {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels),
                 "image_id": r["id"]})


def collate_pairs(batch):
    vis = torch.stack([b[0] for b in batch])
    ir = torch.stack([b[1] for b in batch])
    return vis, ir, [b[2] for b in batch]


def summarize(domain: str) -> dict:
    idx = load_index(domain)
    out = {"domain": domain, "n_images": len(idx["images"]), "classes": idx["classes"]}
    per_split, per_cls, scenes = {}, {}, set()
    for r in idx["images"]:
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1
        scenes.add(r["scene"])
        for l in r["labels"]:
            per_cls[idx["classes"][l]] = per_cls.get(idx["classes"][l], 0) + 1
    out.update(per_split=per_split, per_class=per_cls, n_scenes=len(scenes),
               n_with_ir16=sum(1 for r in idx["images"] if r.get("ir16")),
               n_with_vis=sum(1 for r in idx["images"] if r.get("vis")))
    return out
