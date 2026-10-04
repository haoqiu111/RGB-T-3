# -*- coding: utf-8 -*-
"""Result figures (palette: grey / light grey / sky / blue / green):
  figC14_gate           distributions of the two statistics of the label-free gate on the test draws
  figC1_main_v8         main comparison including the full system (gated inversion + selection), vertical layout
  figC2_per_kind_v8     per-degradation gain: selection only vs full system
  figC4_tau_frontier_v8 margin frontier, logarithmic x axis
Usage: python make_figs_c5.py [gate margin main kind main_h]  (default: all; main / kind / main_h need the inv_blind runs)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import _bootstrap  # noqa: F401,E402

import c_v8_data as D  # noqa: E402

RUNS = D.RUNS; OUT = RUNS / "figures_c"
C = {"grey": "#7F7F7F", "lgrey": "#B8B8B8", "sky": "#56B4E9", "blue": "#0072B2", "navy": "#003F6B", "green": "#009E73", "orange": "#D55E00"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "savefig.bbox": "tight"})
EVT = [("dom_retinanet", "RetinaNet"), ("dom_rtdetr", "RT-DETR-L"), ("dom_yolo", "YOLO11m")]
EPS1, EPS2 = 0.02, 0.10


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=600); plt.close(fig); print("saved", name)


def fig_gate():
    rows = json.loads((RUNS / "c_invert" / "gate_dist.json").read_text(encoding="utf-8"))
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append((r["resid"], r["dev"]))
    by = {k: np.array(v) for k, v in by.items()}
    vis = np.concatenate([by[k] for k in by if k.startswith("vis_")])
    G = [("Clean, as shipped", by["clean"], C["grey"]), ("VIS-side stress, TIR untouched", vis, C["grey"]),
         ("Real AGC: stale window", by["real_agc_stale"], C["blue"]), ("Real AGC: histogram eq.", by["real_agc_histeq"], C["blue"]), ("Real AGC: plateau eq.", by["real_agc_plateau"], C["blue"]),
         ("Real AGC: min-max", by["real_agc_minmax"], C["blue"]), ("Real AGC: gamma", by["real_agc_gamma"], C["blue"]), ("IR low contrast", by["ir_low_contrast"], C["sky"]),
         ("Registration shift", by["misalignment"], C["orange"]), ("IR dead pixels", by["ir_dead_pixels"], C["orange"]), ("IR gain noise", by["ir_gauss_noise"], C["orange"]),
         ("IR stripe NUC", by["ir_stripe_nuc"], C["orange"]), ("IR local dropout", by["ir_region_drop"], C["orange"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.08})
    y = np.arange(len(G))[::-1]

    def boxes(ax, col, items, ypos):
        for (name, a, c), yy in zip(items, ypos):
            v = a[:, col]
            bp = ax.boxplot([v], positions=[yy], vert=False, widths=0.62, whis=(1, 99), patch_artist=True, showfliers=True,
                            flierprops=dict(marker="o", ms=1.6, mfc=c, mec="none", alpha=0.7), medianprops=dict(color="white", lw=1.0),
                            boxprops=dict(facecolor=c, edgecolor=c, lw=0.6), whiskerprops=dict(color=c, lw=0.8), capprops=dict(color=c, lw=0.8))
    ax = axes[0]; boxes(ax, 0, G, y); ax.set_xscale("log"); ax.set_xlim(1e-3, 1.6)
    ax.axvline(EPS1, color=C["navy"], lw=1.0, ls="--"); ax.text(EPS1 * 1.08, len(G) - 0.45, r"$\varepsilon_1$ = 0.02", color=C["navy"], fontsize=7, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([g[0] for g in G], fontsize=7); ax.set_ylim(-0.7, len(G) + 0.1)
    ax.set_xlabel("monotone-fit residual $r$"); ax.set_title("(a) Level one: is the frame a remap of its counts?", fontsize=8, loc="left")
    for (name, a, c), yy in zip(G, y):
        ax.text(1.45, yy, f"{100 * (a[:, 0] <= EPS1).mean():.0f}%", fontsize=6.3, va="center", ha="right", color="#333")
    ax.text(1.45, len(G) - 0.45, "pass", fontsize=6.3, ha="right", va="bottom", color="#333")
    ax = axes[1]; boxes(ax, 1, G[:8], y[:8]); ax.set_xlim(0, 0.78)
    ax.axvline(EPS2, color=C["navy"], lw=1.0, ls="--"); ax.text(EPS2 + 0.008, len(G) - 0.45, r"$\varepsilon_2$ = 0.10", color=C["navy"], fontsize=7, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(-0.7, len(G) + 0.1)
    ax.set_xlabel("deviation $d$ from the learned camera map"); ax.set_title("(b) Level two: did the gain control change?", fontsize=8, loc="left")
    for (name, a, c), yy in zip(G, y):
        fire = ((a[:, 0] <= EPS1) & (a[:, 1] >= EPS2)).mean()
        ax.text(0.77, yy, f"{100 * fire:.0f}%", fontsize=6.3, va="center", ha="right", color="#333")
        if yy < y[7]:
            ax.text(0.12, yy, "rejected at level one", fontsize=6.3, va="center", color=C["orange"], style="italic")
    ax.text(0.77, len(G) - 0.45, "inverted", fontsize=6.3, ha="right", va="bottom", color="#333")
    save(fig, "figC14_gate")


def fig_margin():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    mk = {"dom_retinanet": ("o", C["sky"]), "dom_rtdetr": ("s", C["navy"]), "dom_yolo": ("^", C["blue"])}
    TAGS = ((0.08, "tau0.08", "τ = 0.08"), (0.04, "tau0.04", "τ = 0.04\ninherited"), (0.0, None, "τ = 0\nselected by critics"))
    xs_all = None
    for ev, title in EVT:
        xs, ys = [], []
        for tau, tag, _ in TAGS:
            s = json.loads((D.LYN / ("test_both_s0" + (f"_{tag}" if tag else "")) / "summary.json").read_text(encoding="utf-8"))[ev]
            xs.append(100 * s["clean"]["switch_rate"]); ys.append(100 * s["stress"]["decision_gain"])
        m, c = mk[ev]; ax.plot(xs, ys, marker=m, color=c, lw=1.1, ms=5, label=title); xs_all = xs
        ax.plot([xs[1]], [ys[1]], marker=m, color=c, ms=8.5, mfc="none", mew=1.1)
    ax.set_xscale("log"); ax.set_xlim(0.7, 48); ax.set_xticks([1, 2, 5, 10, 20, 40]); ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.5)
    for x, (_, _, lab) in zip(xs_all, TAGS):
        ax.axvline(x, color=C["lgrey"], lw=0.7, ls=":", zorder=0); ax.text(x, hi + 0.45, lab, fontsize=6.5, ha="center", va="top", color="#333")
    ax.set_xlabel("clean switch rate (%), log scale"); ax.set_ylabel("stress decision gain (points)")
    ax.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    save(fig, "figC4_tau_frontier_v8")


def fig_main():
    S = {t: D.summary(t) for t in ("tau0.04_extra", "sota2", "sota3")}
    PUB = [("tardal", "TarDAL", "tau0.04_extra"), ("seafusion", "SeAFusion", "tau0.04_extra"), ("cddfuse", "CDDFuse", "sota2"), ("emma", "EMMA", "sota2"), ("drmf", "DRMF", "sota3"),
           ("textif", "Text-IF", "sota2"), ("textif_p", "Text-IF + prompt", "sota2"), ("controlfusion", "ControlFusion", "sota3")]
    IND = [("avg", "Average", "tau0.04_extra"), ("max", "Maximum", "tau0.04_extra"), ("gp", "GP-trained", "tau0.04_extra")]
    labels = [l for _, l, _ in PUB + IND] + ["Joint", "Ours, stitched", "No gate, decision", "Ours, decision", "Late fusion"]
    cols = [C["grey"]] * len(PUB) + [C["lgrey"]] * (len(IND) + 1) + [C["sky"], "#A9D8F5", C["blue"], C["green"]]
    fig, axes = plt.subplots(3, 1, figsize=(5.9, 7.9), sharex=True); fig.subplots_adjust(top=0.9, hspace=0.3)
    for ax, (ev, title) in zip(axes, EVT):
        bm = D.block_means(ev); st, cl, nfr = [], [], []
        for c, _, t in PUB + IND:
            e = D.extra_means(t, c, ev); st.append(e["stress"]["q"]); cl.append(e["clean"]["q"]); nfr.append(e["stress"]["nfr"])
        for key in ("fused", "full_stitched", "sel_decision", "full_decision", "late"):
            st.append(bm["stress"][key]); cl.append(bm["clean"][key]); nfr.append(bm["stress"]["nfr_" + key])
        best = bm["stress"]["best"]; x = np.arange(len(labels))
        ax.bar(x, st, color=cols, width=0.72)
        ax.scatter(x, cl, facecolor="white", edgecolor="#222222", s=14, zorder=4, lw=0.8)
        ax.axhline(best, color=C["navy"], lw=1.0, ls="--", zorder=1)
        for i, n in enumerate(nfr):
            ax.text(i, 0.105, f"{100 * n:.0f}", ha="center", va="bottom", fontsize=5.6, color="#222222" if cols[i] in (C["lgrey"], "#A9D8F5", C["sky"]) else "white")
        top = max(cl + [best]) + 0.035; ax.set_ylim(0.10, top)
        ax.text(0.01, best + 0.004, "best-single oracle, stress", color=C["navy"], fontsize=6.5, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_ylabel("Evidence quality"); ax.set_title(f"{title} evaluator", fontsize=8.5, loc="right")
    axes[-1].set_xticks(range(len(labels))); axes[-1].set_xticklabels(labels, fontsize=7, rotation=40, ha="right")
    axes[0].legend(handles=[Patch(color=C["grey"], label="published fusion, zero-shot"), Patch(color=C["lgrey"], label="in-domain always-fuse"),
                            Patch(color=C["sky"], label="ours, stitched, one pass"), Patch(color="#A9D8F5", label="ours without the gate, decision-level"), Patch(color=C["blue"], label="ours, decision-level"),
                            Patch(color=C["green"], label="late fusion, two passes"), Line2D([], [], marker="o", mfc="white", mec="#222", ls="", ms=4, label="quality on clean data"),
                            Line2D([], [], ls="", marker="$47$", mec="#444", mfc="#444", ms=7, label="in bars: NFR under stress (%)")],
                   fontsize=6.2, ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.08), columnspacing=1.4, handletextpad=0.6)
    save(fig, "figC1_main_v8")


def fig_main_h():
    """Main comparison in a 1x3 layout (one panel per evaluator), slanted system names, legend in one block on top."""
    S = {t: D.summary(t) for t in ("tau0.04_extra", "sota2", "sota3")}
    PUB = [("tardal", "TarDAL", "tau0.04_extra"), ("seafusion", "SeAFusion", "tau0.04_extra"), ("cddfuse", "CDDFuse", "sota2"), ("emma", "EMMA", "sota2"), ("drmf", "DRMF", "sota3"),
           ("textif", "Text-IF", "sota2"), ("textif_p", "Text-IF + prompt", "sota2"), ("controlfusion", "ControlFusion", "sota3")]
    labels = [l for _, l, _ in PUB] + ["Joint (in-domain)", "Ours, stitched", "No gate, decision", "Ours, decision", "Late fusion"]
    cols = [C["grey"]] * len(PUB) + [C["lgrey"], C["sky"], "#A9D8F5", C["blue"], C["green"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), sharey=False); fig.subplots_adjust(top=0.8, wspace=0.22, bottom=0.3, left=0.06, right=0.995)
    for ax, (ev, title) in zip(axes, EVT):
        bm = D.block_means(ev); st, cl, nfr = [], [], []
        for c, _, t in PUB:
            e = D.extra_means(t, c, ev); st.append(e["stress"]["q"]); cl.append(e["clean"]["q"]); nfr.append(e["stress"]["nfr"])
        for key in ("fused", "full_stitched", "sel_decision", "full_decision", "late"):
            st.append(bm["stress"][key]); cl.append(bm["clean"][key]); nfr.append(bm["stress"]["nfr_" + key])
        best = bm["stress"]["best"]; x = np.arange(len(labels))
        ax.bar(x, st, color=cols, width=0.72)
        ax.scatter(x, cl, facecolor="white", edgecolor="#222222", s=13, zorder=4, lw=0.8)
        ax.axhline(best, color=C["navy"], lw=1.0, ls="--", zorder=1)
        for i, n in enumerate(nfr):
            ax.text(i, 0.105, f"{100 * n:.0f}", ha="center", va="bottom", fontsize=5.4, color="#222222" if cols[i] in (C["lgrey"], "#A9D8F5", C["sky"]) else "white")
        top = max(cl + [best]) + 0.03; ax.set_ylim(0.10, top)
        ax.text(0.01, best + 0.003, "best-single oracle, stress", color=C["navy"], fontsize=6.2, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_title(f"{title} evaluator", fontsize=8.5, loc="right"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6, rotation=45, ha="right")
    axes[0].set_ylabel("Evidence quality")
    fig.legend(handles=[Patch(color=C["grey"], label="published fusion, zero-shot"), Patch(color=C["lgrey"], label="in-domain always-fuse"),
                        Patch(color=C["sky"], label="ours, stitched, one pass"), Patch(color="#A9D8F5", label="ours without the gate, decision-level"), Patch(color=C["blue"], label="ours, decision-level"),
                        Patch(color=C["green"], label="late fusion, two passes"), Line2D([], [], marker="o", mfc="white", mec="#222", ls="", ms=4, label="quality on clean data"),
                        Line2D([], [], ls="", marker="$47$", mec="#444", mfc="#444", ms=7, label="in bars: NFR under stress (%)")],
               fontsize=6.6, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handletextpad=0.6)
    save(fig, "figC1_main_v9")


def fig_kind():
    kinds = D.SYN + D.REAL
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    for ax, (ev, title) in zip(axes, EVT):
        pk = D.per_kind(ev); y = np.arange(len(kinds))
        o = [100 * (pk[k]["best"] - pk[k]["fused"]) for k in kinds]; gs = [100 * (pk[k]["sel"] - pk[k]["fused"]) for k in kinds]; gf = [100 * (pk[k]["full"] - pk[k]["fused"]) for k in kinds]
        ax.barh(y, o, color=C["lgrey"], height=0.78)
        ax.barh(y - 0.17, gs, color=C["sky"], height=0.3); ax.barh(y + 0.17, gf, color=C["blue"], height=0.3)
        for i, v in enumerate(gf):
            ax.text(max(v, o[i], 0) + 0.4, i, f"{v:+.1f}", va="center", fontsize=6.3)
        ax.set_yticks(y); ax.set_yticklabels([D.KNAME[k] for k in kinds], fontsize=7); ax.invert_yaxis(); ax.set_title(f"{title} evaluator", fontsize=9)
        ax.axvline(0, color="#333", lw=0.6); ax.set_xlabel("evidence points over always-fuse"); ax.axhline(len(D.SYN) - 0.5, color=C["grey"], lw=0.7, ls=":")
        ax.set_xlim(min(ax.get_xlim()[0], -1), ax.get_xlim()[1] * 1.08)
    fig.legend(handles=[Patch(color=C["lgrey"], label="oracle gap, best single modality minus always-fuse"), Patch(color=C["sky"], label="selection only"),
                        Patch(color=C["blue"], label="full system, gated inversion and selection (value printed)")],
               fontsize=7.2, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.045))
    save(fig, "figC2_per_kind_v8")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["gate", "margin", "main", "kind", "main_h"]
    for name, fn in (("gate", fig_gate), ("margin", fig_margin), ("main", fig_main), ("kind", fig_kind), ("main_h", fig_main_h)):
        if name in todo:
            fn()
