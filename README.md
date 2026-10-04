# Selective Visible-Infrared Fusion under Real Thermal Gain Control

Code for the paper

> Jun Yao, Zhilin Guo, Jiajun Song. *Selective Visible-Infrared Fusion under Real Thermal Gain Control.* Submitted to Expert Systems with Applications.

Visible-infrared fusion is usually deployed as an unconditional contract: every input pair yields one fused image that the detector must accept. This repository implements a selective alternative and the protocol used to evaluate it on a real camera:

* **Real gain-control stress.** Five automatic gain control (AGC) families — histogram equalization, plateau equalization, gamma, min-max stretch and a stale window — are executed on the 16-bit raw thermal counts of LYNRED-MDS, so the thermal frame is exactly what a differently configured camera emits (`cdkit/thermal16.py`, `cdkit/realdeg.py`).
* **Label-free gated inversion.** A two-level gate on the raw counts (`cdkit/ispgate.py`, mode `gate2b`) fits a monotone map from counts to the 8-bit frame, rejects frames that are not a pure tone remap (sensor faults), compares the frame with a learned camera tone map, and re-executes that map before fusion.
* **Region-level selection.** A frozen fusion network, a fingerprint quality-regression router and a margin rule choose FUSE, VIS or TIR per region; conformal risk control accepts or abstains.
* **Evaluation.** Three hold-out detectors (RetinaNet, RT-DETR-L, YOLO11m) that never saw a fused image, six published fusion models run zero-shot with their official code and weights, decision-level late fusion as a reference, COCO metrics, a counterfactual evidence audit, and zero-retraining deployment on further visible-infrared datasets.

All reported numbers are produced by the scripts below; `scripts/c_v8_data.py` is the single module that turns the run outputs into the values of the tables and figures.

## Repository layout

```
cdkit/                     dataset index, paired loader, 16-bit thermal processing, real AGC menu, gated inversion
  paths.py                 every filesystem location, configurable through environment variables
  index.py                 unified index and loader: vis[3,H,W] ir[1,H,W] boxes labels image_id scene_id
  thermal16.py             16-bit raw counts -> 8-bit tone maps (canonical stretch and AGC families)
  realdeg.py               stress menu: synthetic degradations + real AGC families on raw counts
  ispgate.py               monotone-fit residual, learned camera map, gate modes (gate, gate2, gate2b, oracle)
scripts/                   training, evaluation, statistics and figure scripts (see REPRODUCE.md)
registry/splits/           frozen train/val/calib/test id lists of the domains
```

## Installation

Tested with Python 3.12 on Windows 11 and an RTX 5090 (CUDA 13.2).

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from https://pytorch.org first if the pinned wheels do not fit your platform.

### Companion code from our previous work

The fusion network (`rgbta.models.fusion_net`), the router (`rgbta.models.router`), the utility and conformal routines (`rgbta.eval`), the synthetic degradation menu (`rgbta.data.degradations`) and several training helpers (`train_fusion_joint.py`, `precompute_region_labels.py`, `train_router.py`, `train_router_v2.py`, `train_critic_detector.py`, `train_heldout_retinanet.py`, `eval_heldout.py`, `eval_sota.py`, `gate_a0.py`) come from the code of our previous selective-fusion work on FLIR, which this paper extends. Point `SFA_RGBT_PROJECT` to that checkout (it must contain the `rgbta/` package and its `scripts/` folder). The class-activation-map figures additionally need `det_cam_util.py`; point `SFA_CTDET_SCRIPTS` to the folder that holds it.

### Third-party fusion models (zero-shot baselines)

Third-party code and weights are not redistributed. Clone the official repositories into `$SFA_THIRD_PARTY` under these folder names and download their released weights:

| Model | Folder under `$SFA_THIRD_PARTY` | Weights used |
|---|---|---|
| CDDFuse (CVPR 2023) | `MMIF-CDDFuse` | `models/CDDFuse_IVF.pth` |
| EMMA (CVPR 2024) | `MMIF-EMMA` | `model/EMMA.pth` |
| Text-IF (CVPR 2024) | `Text-IF` | `pretrained_weights/simple_fusion.pth`, `pretrained_weights/text_fusion_power.pth` |
| ControlFusion (NeurIPS 2025) | `ControlFusion` | `$SFA_WEIGHTS/fusion_weights/pretrained_weights/` |
| DRMF (ACM MM 2024) | `DRMF` | `$SFA_WEIGHTS/fusion_weights/experiment/` |

TarDAL and SeAFusion are loaded through `eval_sota.py` of the companion code. The exact checkpoint file names and preprocessing are documented in the docstrings of `scripts/c_sota2.py`. Every third-party checkpoint is loaded with `weights_only=True`.

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `SFA_DATA_ROOT` | extracted datasets | `<repo>/data/extracted` |
| `SFA_REGISTRY` | index and split descriptors | `<repo>/registry` |
| `SFA_RUNS` | checkpoints, labels, reports, figures | `<repo>/runs` |
| `SFA_WEIGHTS` | external weights | `<repo>/weights` |
| `SFA_THIRD_PARTY` | official fusion-model checkouts | `<repo>/third_party` |
| `SFA_RGBT_PROJECT` | companion code (`rgbta` package + scripts) | `<repo>/external/rgbt_project` |
| `SFA_CTDET_SCRIPTS` | folder with `det_cam_util.py` (CAM figures only) | `<repo>/external/ctdet_project/scripts` |

Every script imports `scripts/_bootstrap.py` first, which puts the repository, `scripts/` and the companion code on `sys.path`, so scripts can be launched from any working directory.

## Data

All datasets are public; download them from their official sources and extract them under `$SFA_DATA_ROOT/<domain>/`. The folder names expected by the index builders are in `scripts/p1_build_index.py`.

| Domain key | Dataset | Role |
|---|---|---|
| `lynred_mds` | LYNRED-MDS (8-bit and 16-bit thermal frames) | main domain, real AGC stress |
| `flir_align` | FLIR-aligned | control domain, second selector |
| `fmb`, `mfad`, `smod`, `rlivit_rgbt`, `m3fd`, `llvip`, `kaist`, `msrs`, `mfnet` | FMB, MFAD, SMOD, R-LiViT, M3FD, LLVIP, KAIST, MSRS, MFNet | cross-camera deployment |
| `rgbt_droneperson`, `dronevehicle` | RGBTDronePerson, DroneVehicle | aerial deployment (reported, not interpreted) |

Build the unified index once, then use the frozen splits shipped in `registry/splits/` (or rebuild them; `p2_build_splits.py` never overwrites an existing split):

```bash
python scripts/p1_build_index.py --domains lynred_mds,fmb,mfad,smod,rlivit_rgbt,msrs,mfnet,rgbt_droneperson,dronevehicle,flir_align,llvip,m3fd,kaist
python scripts/p2_build_splits.py
python scripts/smoke_domains.py --domains lynred_mds --n 3
```

Splits of FLIR-aligned, LLVIP and M3FD are taken from the companion project.

## Reproducing the paper

`REPRODUCE.md` lists every command, in order, with the flags used for the paper. In short:

```bash
# models on LYNRED-MDS
python scripts/c_train_critic.py --domain lynred_mds --modality vis --epochs 6
python scripts/c_train_critic.py --domain lynred_mds --modality ir --epochs 6
python scripts/c_train_fusion.py --domain lynred_mds --stage gp --epochs 8
python scripts/c_train_fusion.py --domain lynred_mds --stage joint --epochs 3
python scripts/c_precompute_labels.py --domain lynred_mds --splits train val --draws 2
python scripts/c_train_router.py --domain lynred_mds --epochs 8 --seed 0
python scripts/c_train_agc_emulator2.py
# full system: label-free gated inversion + selection, three hold-out evaluators
python scripts/c_eval.py --system lynred_mds --seed 0 --domain lynred_mds --split test --draws 3 --menu both \
  --evaluators dom_retinanet,dom_rtdetr,dom_yolo --tau 0.04 --extra late \
  --invert gate2b --canon emulator --gate2-thr 0.10 --tag inv_blind
```

## Citation

```bibtex
@article{yao2026selective,
  title   = {Selective Visible-Infrared Fusion under Real Thermal Gain Control},
  author  = {Yao, Jun and Guo, Zhilin and Song, Jiajun},
  journal = {Expert Systems with Applications},
  note    = {Under review},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`. Datasets and third-party models keep their own licenses.
