# Reproducing the paper

All commands run from the repository root with the environment variables of the README set. Outputs go to `$SFA_RUNS`. Every command below is the one used for the paper. Times are for one RTX 5090.

Common evaluator list: `EV=dom_retinanet,dom_rtdetr,dom_yolo`.

## 1. Hold-out evaluators (never trained on fused frames)

```bash
python scripts/train_heldout_domain.py --domain lynred_mds --classes super --epochs 8
python scripts/train_heldout_domain.py --domain fmb --classes super --epochs 10
python scripts/train_heldout_domain.py --domain mfad --classes super --epochs 6 --max-train 4000
python scripts/train_heldout_domain.py --domain smod --classes super --epochs 6 --max-train 4000
python scripts/train_heldout_domain.py --domain rlivit_rgbt --classes super --epochs 8
python scripts/c_prep_yolo_data.py --domain lynred_mds
python scripts/c_train_ultra.py --domain lynred_mds --arch rtdetr --epochs 20
python scripts/c_train_ultra.py --domain lynred_mds --arch yolo --epochs 20
```

## 2. Critics, fusion network, router, camera map

```bash
python scripts/c_train_critic.py --domain lynred_mds --modality vis --epochs 6
python scripts/c_train_critic.py --domain lynred_mds --modality ir --epochs 6
python scripts/c_train_fusion.py --domain lynred_mds --stage gp --epochs 8
python scripts/c_train_fusion.py --domain lynred_mds --stage joint --epochs 3
python scripts/c_precompute_labels.py --domain lynred_mds --splits train val --draws 2
for s in 0 1 2; do python scripts/c_train_router.py --domain lynred_mds --epochs 8 --seed $s; done
for s in 0 1 2; do python scripts/c_select_tau.py --domain lynred_mds --seed $s; done
python scripts/c_train_agc_emulator2.py            # learned camera tone map (level two of the gate)
python scripts/c_isp_gate_calib.py --limit 150      # monotone-fit residual statistics on the training split
```

The margin used in the paper is inherited from the FLIR system, `--tau 0.04`; `c_select_tau.py` gives the critic-selected margin reported in the margin ablation.

## 3. Main comparison on LYNRED-MDS (Tables 1-3, 6, Figures 3-4)

```bash
BASE="--system lynred_mds --domain lynred_mds --split test --draws 3 --menu both --evaluators $EV --tau 0.04"
INV="--invert gate2b --canon emulator --gate2-thr 0.10"
# selection only, with in-domain always-fuse baselines and late fusion
python scripts/c_eval.py $BASE --seed 0 --extra avg,max,gp,tardal,seafusion,late --tag tau0.04_extra
python scripts/c_eval.py $BASE --seed 1 --tag tau0.04
python scripts/c_eval.py $BASE --seed 2 --tag tau0.04
# full system: label-free gated inversion + selection, three seeds
for s in 0 1 2; do python scripts/c_eval.py $BASE --seed $s --extra late $INV --tag inv_blind; done
# published fusion models, zero-shot
python scripts/c_eval.py $BASE --seed 0 --extra cddfuse,emma,textif,textif_p,late --tag sota2
python scripts/c_eval.py $BASE --seed 0 --extra controlfusion,controlfusion_p,drmf --tag sota3
# selector over ControlFusion in place of the joint network
python scripts/c_eval.py $BASE --seed 0 --extra controlfusion,textif_agc,late --fuse-backbone controlfusion --tag cfsel
# gate ablation: oracle inversion and single-level variants
python scripts/c_eval.py $BASE --seed 0 --extra late --invert oracle --canon emulator --tag inv_emu_oracle
python scripts/c_eval.py $BASE --seed 0 --extra late --invert gate --tag inv_gate
python scripts/c_eval.py $BASE --seed 0 --extra late --invert gate --canon emulator --tag inv_emu
# image-level descriptor gate baseline
python scripts/c_descriptor_gate.py --domain lynred_mds --max-images 1200
python scripts/c_eval.py $BASE --seed 0 --extra dgate,late --tag dgate
# margin sweep
python scripts/c_eval.py $BASE --seed 0 --tau 0.08 --tag tau0.08
# confidence intervals over scene groups of 50 frames
python scripts/c_stats.py --seeds 0,1,2 --group-frames 50
python scripts/c_gate_dist.py                         # gate statistics on the test draws (Figure 2)
```

`c_v8_data.py` aligns the gated and ungated runs box by box and reports the gains of the full system against always-fuse and best-single on the frames as emitted.

## 4. Cross-camera deployment (Table 4)

For each `DOM:EVALUATORS:SUBSAMPLE` in

```
fmb:dom_retinanet,flir_rtdetr:0   mfad:dom_retinanet,flir_rtdetr:800   smod:dom_retinanet,flir_rtdetr:800
rlivit_rgbt:dom_retinanet,flir_rtdetr:0   flir_align:flir_retinanet,flir_rtdetr,flir_yolo:0
m3fd:flir_retinanet,flir_rtdetr:800   llvip:flir_retinanet,flir_rtdetr:800   kaist:flir_retinanet,flir_rtdetr:800
msrs:flir_retinanet,flir_rtdetr:0   mfnet:flir_retinanet,flir_rtdetr:0
rgbt_droneperson:flir_retinanet,flir_rtdetr:600   dronevehicle:flir_retinanet,flir_rtdetr:600
```

and each system in `lynred_mds flir`:

```bash
python scripts/c_eval.py --system $SYSTEM --domain $DOM --split test --draws 3 --menu both --evaluators $EVALUATORS --subsample $SUBSAMPLE --tau 0.04
```

M3FD uses `--split eval`.

## 5. Cost (Table 5)

```bash
python scripts/c_cost.py
python scripts/c_cost_sota.py
python scripts/c_cost_gate.py     # CPU timing of the gate; output identical to the experiment path
```

## 6. Risk control, controls and audits (Tables 7-8, supplementary)

```bash
for cfg in "exchangeable group 0" "exchangeable group 50" "exchangeable box 0" "shifted group 50" "shifted box 0"; do
  set -- $cfg
  python scripts/c_calibrate.py --domain lynred_mds --seed 0 --protocol $1 --unit $2 --menu both --group-frames $3
done
# evaluator trained on fused frames
python scripts/c_precompute_fused.py --domain lynred_mds
python scripts/train_heldout_domain.py --domain lynred_mds --classes super --epochs 8 --fused-dir $SFA_RUNS/fused_imgs/lynred_mds --tag fused
python scripts/c_eval.py $BASE --seed 0 --evaluators dom_retinanet_fused --extra avg,max,gp,tardal,seafusion,late --tag evfused
# COCO metrics
COCO="--system lynred_mds --seed 0 --domain lynred_mds --tau 0.04 --evaluators $EV --draws 3"
python scripts/c_eval_coco.py $COCO
python scripts/c_eval_coco.py $COCO $INV --tag inv_blind
# late fusion as a frame-level fourth action
python scripts/c_late_policy.py --run test_both_s0_tau0.04_extra
# counterfactual evidence audit and temporal stability
python scripts/c_counterfactual.py --limit 400 --select thermal_dep
python scripts/c_temporal.py --frames 240 --stride 2
```

## 7. Figures

```bash
python scripts/make_figs_c5.py            # gate statistics, main comparison, per-degradation gain, margin frontier
python scripts/make_qual_c3.py            # qualitative and class-activation-map figures
```
