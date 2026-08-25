"""Generate paper figures from *_final_* result JSONs into docs/figures/."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load_final(prefix):
    files = sorted(RES.glob(f"{prefix}_final_*.json"))
    return json.loads(files[-1].read_text())


def errbar(ax, labels, means, cis, color):
    ax.bar(labels, means, yerr=cis, capsize=3, color=color,
           edgecolor="black", linewidth=0.4)
    ax.set_ylim(0, 1.05)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)


def fig_baselines():
    base = load_final("baselines")["summary"]
    hard = load_final("hard_conditions_final".replace("_final", "")) if False else None
    hf = sorted(RES.glob("hard_conditions_final_*.json"))[-1]
    hard = json.loads(hf.read_text())["summary"]
    rnd_e = json.loads((RES / "base_random_1_30.json").read_text())
    rnd_h = json.loads((RES / "hard_random_1_30.json").read_text())

    def add_random(summary, rnd):
        comps = [r["completion_rate"] for r in rnd["results"]]
        summary = dict(summary)
        summary["random"] = {"completion_rate": {
            "mean": float(np.mean(comps)),
            "ci95": float(1.96*np.std(comps, ddof=1)/np.sqrt(len(comps)))}}
        return summary

    base_s = add_random(base, rnd_e)
    hard_s = add_random(hard, rnd_h)
    order = ["random", "distance", "astar", "standard_aco", "full_cms_daco"]
    labels = ["Random", "Distance\n(BFS)", "Hazard-aware\nA*", "Standard\nACO", "Full\nCMS-DACO"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, summ, title in [(axes[0], base_s, "Open grid"), (axes[1], hard_s,
                            "Hard: 0.15 walls, blocked exit, wind")]:
        means = [summ[m]["completion_rate"]["mean"] for m in order]
        cis = [summ[m]["completion_rate"]["ci95"] or 0 for m in order]
        errbar(ax, labels, means, cis, "#72a0d6")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Evacuation completion" if ax is axes[0] else "")
    fig.tight_layout()
    fig.savefig(OUT / "fig_baselines.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    abl = load_final("ablation")["summary"]
    order = ["full", "wo_bfs_seed", "wo_dual", "wo_predictive", "wo_escape", "wo_hazard_aware"]
    labels = ["Full", "w/o BFS\nseed", "w/o dual\nchannel", "w/o pred.\ncongestion",
              "w/o escape", "w/o hazard\npenalty"]
    means = [abl[m]["completion_rate"]["mean"] for m in order]
    cis = [abl[m]["completion_rate"]["ci95"] or 0 for m in order]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    errbar(ax, labels, means, cis, "#8fbf7f")
    ax.set_ylabel("Evacuation completion")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_robustness():
    rob = load_final("robustness")["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    # walls
    w = sorted(rob["walls"].items(), key=lambda kv: float(kv[0]))
    x = [float(k) for k, _ in w]
    m = [v["completion_mean"] for _, v in w]
    c = [v["completion_ci"] for _, v in w]
    axes[0].errorbar(x, m, yerr=c, marker="o", capsize=3, color="#c0504d")
    axes[0].set_xlabel("Wall density"); axes[0].set_ylabel("Completion")
    axes[0].set_ylim(0, 1.05); axes[0].set_title("Obstacle density", fontsize=9)
    # crowd
    cw = sorted(rob["crowd"].items(), key=lambda kv: float(kv[0]))
    x = [float(k) for k, _ in cw]
    m = [v["completion_mean"] for _, v in cw]
    c = [v["completion_ci"] for _, v in cw]
    axes[1].errorbar(x, m, yerr=c, marker="s", capsize=3, color="#4f81bd")
    axes[1].set_xlabel("Agents"); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Crowd size", fontsize=9)
    # wind
    wd = sorted(rob["wind"].items(), key=lambda kv: kv[0])
    labels = [k.replace("_", " ") for k, _ in wd]
    m = [v["completion_mean"] for _, v in wd]
    c = [v["completion_ci"] for _, v in wd]
    axes[2].bar(labels, m, yerr=c, capsize=3, color="#9bbb59",
                edgecolor="black", linewidth=0.4)
    axes[2].set_ylim(0, 1.05); axes[2].set_title("Wind", fontsize=9)
    for tick in axes[2].get_xticklabels():
        tick.set_rotation(15)
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity():
    sens = load_final("sensitivity")["summary"]
    params = [("ALPHA", r"$\alpha$ (pheromone weight)"),
              ("BETA", r"$\beta$ (heuristic weight)"),
              ("RHO", r"$\rho$ (evaporation)"),
              ("ACO_TEMPERATURE", r"$T$ (softmax temperature)")]
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.6))
    for ax, (key, label) in zip(axes.flat, params):
        items = sorted(sens[key].items(), key=lambda kv: float(kv[0]))
        x = [float(k) for k, _ in items]
        m = [v["completion_mean"] for _, v in items]
        c = [v["completion_ci"] for _, v in items]
        ax.errorbar(x, m, yerr=c, marker="o", ms=3, capsize=2, color="#8064a2")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylim(0.85, 1.01)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Completion", fontsize=8)
    axes[1, 0].set_ylabel("Completion", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_baselines()
    fig_ablation()
    fig_robustness()
    fig_sensitivity()
    print("Figures written to", OUT)
