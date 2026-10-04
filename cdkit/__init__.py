# -*- coding: utf-8 -*-
"""Toolkit for selective visible-infrared fusion under real thermal gain control.

Cross-dataset unified index, a common paired loader and 16-bit thermal processing.
The fusion network, router, utility metrics and synthetic degradation menu come
from the companion package `rgbta` (see README); this package only maps every
dataset onto one protocol: vis[3,H,W] ir[1,H,W] boxes(xyxy) labels image_id scene_id.
"""
from pathlib import Path

from cdkit.paths import CTDET_SCRIPTS, DATA_ROOT, REGISTRY, RGBT_PROJECT, ROOT, RUNS  # noqa: F401

ROOT = Path(ROOT)
EXT = DATA_ROOT
REG = REGISTRY
INDEX = REG / "index"
SPLITS = REG / "splits"
RUNS = Path(RUNS)
RGBT_PROJECT = str(RGBT_PROJECT)
CTDET_PROJECT = str(Path(CTDET_SCRIPTS).parent)

# Super-class ids shared with the FLIR-trained system: 0=person 1=car 2=bicycle
SUPER_IDS = {"person": 0, "car": 1, "bicycle": 2}
# native class name -> super-class name; classes not listed are dropped in 'super' mode
SUPER_SYNONYMS = {
    "person": "person", "people": "person", "pedestrian": "person",
    "survivor": "person", "rider": "person", "crowd": "person",
    "car": "car", "van": "car",
    "bicycle": "bicycle", "bike": "bicycle", "cyclist": "bicycle",
}
