# -*- coding: utf-8 -*-
"""Filesystem configuration for the release.

Every location is read from an environment variable and falls back to a path
relative to the repository root, so nothing in the code base refers to a
particular machine.

  SFA_ROOT           repository root            (default: parent of this package)
  SFA_DATA_ROOT      extracted datasets         (default: <root>/data/extracted)
  SFA_REGISTRY       index / split descriptors  (default: <root>/registry)
  SFA_RUNS           checkpoints, labels, reports, figures (default: <root>/runs)
  SFA_WEIGHTS        external weights not shipped inside a third-party checkout
                     (default: <root>/weights)
  SFA_THIRD_PARTY    official fusion-model checkouts (default: <root>/third_party)
  SFA_RGBT_PROJECT   root of the companion project that provides the `rgbta`
                     package and its training/evaluation scripts
                     (default: <root>/external/rgbt_project)
  SFA_CTDET_SCRIPTS  directory holding det_cam_util.py (CAM figures only)
                     (default: <root>/external/ctdet_project/scripts)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SFA_ROOT", Path(__file__).resolve().parents[1]))


def _p(var, default):
    return Path(os.environ.get(var, default))


DATA_ROOT = _p("SFA_DATA_ROOT", ROOT / "data" / "extracted")
REGISTRY = _p("SFA_REGISTRY", ROOT / "registry")
RUNS = _p("SFA_RUNS", ROOT / "runs")
WEIGHTS = _p("SFA_WEIGHTS", ROOT / "weights")
THIRD_PARTY = _p("SFA_THIRD_PARTY", ROOT / "third_party")
RGBT_PROJECT = _p("SFA_RGBT_PROJECT", ROOT / "external" / "rgbt_project")
CTDET_SCRIPTS = _p("SFA_CTDET_SCRIPTS", ROOT / "external" / "ctdet_project" / "scripts")
SCRIPTS = ROOT / "scripts"


def add_sys_paths():
    """Put the repository root, scripts/ and the external companion code on sys.path.

    Insertion order reproduces the original precedence: the companion project's
    scripts come first, then the companion package root, then this repository.
    Idempotent."""
    import sys
    for p in (ROOT, SCRIPTS, CTDET_SCRIPTS, RGBT_PROJECT, RGBT_PROJECT / "scripts"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
