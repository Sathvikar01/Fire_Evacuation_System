"""Generate figures for private observability and scale experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "cms" / "experiments" / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def latest(prefix):
    files = sorted(RES.glob(f"{prefix}_final_*.json"))
    return json.loads(files[-1].read_text())


def observability():
    data = latest("observability")["summary"]
    astar = data["astar"]
    daco = data["full_cms_daco"]
    modes = ["full", "r3", "stale5", "loss50", "priv3"]
    labels = ["Full", "Shared\nr=3", "Stale\n5 ticks", "50% loss", "Private\nr=3"]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.arange(len(modes))
    for key, label, color, marker in [(astar, "A*", "#c0504d", "o"),
                                      (daco, "Full CMS-DACO", "#4f81bd", "s")]:
        y = [key[m]["completion_mean"] for m in modes]
        e = [key[m]["completion_ci"] for m in modes]
        ax.errorbar(x, y, yerr=e, marker=marker, capsize=3, label=label,
                    color=color, linewidth=1.2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Completion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_observability_private.pdf", bbox_inches="tight")
    plt.close(fig)


def scale():
    data = latest("scale")["summary"]
    policies = ["distance", "astar", "dstar", "full_cms_daco"]
    labels = ["Distance", "A*", "D* Lite", "Full DACO"]
    colors = ["#55a868", "#c0504d", "#8172b2", "#4c72b0"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    x = np.arange(2)
    for p, label, color in zip(policies, labels, colors):
        y = [data[e][p]["completion_mean"] for e in ["S30", "S40"]]
        e = [data[e][p]["completion_ci"] for e in ["S30", "S40"]]
        t = [data[e][p]["sec_per_run_mean"] for e in ["S30", "S40"]]
        axes[0].errorbar(x, y, yerr=e, marker="o", capsize=3,
                         label=label, color=color)
        axes[1].plot(x, t, marker="o", label=label, color=color)
    axes[0].set_xticks(x, ["30x30", "40x40"])
    axes[0].set_ylim(0.7, 1.02)
    axes[0].set_ylabel("Completion")
    axes[1].set_xticks(x, ["30x30", "40x40"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Seconds per run (log)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scale_runtime.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    observability()
    scale()
    print(f"Wrote figures to {OUT}")
