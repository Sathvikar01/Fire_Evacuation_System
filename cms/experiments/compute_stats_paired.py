"""Paired statistics for seeded cross-policy comparisons.

Runs share seeds (same layout/fire/spawn per seed index), so observations are
paired. We report paired t-tests, Wilcoxon signed-rank (robustness), and
percentile-bootstrap CIs of the mean paired difference, plus Holm-Bonferroni
correction within each comparison family.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).parent / "results"
RNG = np.random.default_rng(20260826)


def load_final(prefix):
    files = sorted(RES.glob(f"{prefix}_final_*.json"))
    return json.loads(files[-1].read_text())


def comps(entry):
    return np.array([r["completion_rate"] for r in entry], dtype=float)


def paired_bootstrap_ci(diff, n_boot=20000, conf=0.95):
    n = len(diff)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1-conf)/2*100, (1+conf)/2*100])
    return float(lo), float(hi)


def paired_compare(a, b):
    """a, b paired samples (same seed index). Returns dict of tests."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    md = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    dz = md / sd if sd > 0 else 0.0            # Cohen's d_z for paired
    t, tp = stats.ttest_rel(a, b)
    try:
        w, wp = stats.wilcoxon(a, b, zero_method="wilcox")
        wp = float(wp)
    except ValueError:                          # all differences zero
        w, wp = 0.0, 1.0
    lo, hi = paired_bootstrap_ci(diff)
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": md, "sd_diff": sd, "cohens_dz": dz,
        "paired_t": float(t), "t_p": float(tp),
        "wilcoxon_W": float(w), "wilcoxon_p": wp,
        "boot_ci95": [lo, hi],
        "n": len(a),
    }


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def compare_family(name, pairs):
    """pairs: list of (label, a_array, b_array). Returns records with Holm."""
    recs = []
    for label, a, b in pairs:
        rec = {"comparison": label}
        rec.update(paired_compare(a, b))
        recs.append(rec)
    ps = [r["t_p"] for r in recs]
    for r, padj in zip(recs, holm(ps)):
        r["holm_p"] = float(padj)
    print(f"\n=== {name} (paired, Holm-corrected) ===")
    for r in recs:
        print(f"{r['comparison']:<34} d={r['mean_diff']:+.4f} "
              f"dz={r['cohens_dz']:+.2f} t={r['paired_t']:+.2f} "
              f"p={r['t_p']:.4g} holm_p={r['holm_p']:.4g} "
              f"boot95=[{r['boot_ci95'][0]:+.4f},{r['boot_ci95'][1]:+.4f}] "
              f"Wilcoxon_p={r['wilcoxon_p']:.4g}")
    return recs


def main():
    base = load_final("baselines")["results"]
    hf = sorted(RES.glob("hard_conditions_final_*.json"))[-1]
    hard = json.loads(hf.read_text())["results"]
    abl = load_final("ablation")["results"]

    rnd_e = comps(json.loads((RES/"base_random_1_30.json").read_text())["results"])
    rnd_h = comps(json.loads((RES/"hard_random_1_30.json").read_text())["results"])

    out = {}
    out["easy"] = compare_family("Easy/open", [
        ("full vs astar", comps(base["full_cms_daco"]), comps(base["astar"])),
        ("full vs distance", comps(base["full_cms_daco"]), comps(base["distance"])),
        ("full vs standard_aco", comps(base["full_cms_daco"]), comps(base["standard_aco"])),
        ("astar vs random", comps(base["astar"]), rnd_e),
    ])
    out["hard"] = compare_family("Hard", [
        ("full vs astar", comps(hard["full_cms_daco"]), comps(hard["astar"])),
        ("full vs distance", comps(hard["full_cms_daco"]), comps(hard["distance"])),
        ("full vs standard_aco", comps(hard["full_cms_daco"]), comps(hard["standard_aco"])),
        ("astar vs random", comps(hard["astar"]), rnd_h),
        ("distance vs random", comps(hard["distance"]), rnd_h),
    ])
    out["ablation"] = compare_family("Ablation", [
        ("full vs wo_bfs_seed", comps(abl["full"]), comps(abl["wo_bfs_seed"])),
        ("full vs wo_dual", comps(abl["full"]), comps(abl["wo_dual"])),
        ("full vs wo_predictive", comps(abl["full"]), comps(abl["wo_predictive"])),
        ("full vs wo_escape", comps(abl["full"]), comps(abl["wo_escape"])),
        ("full vs wo_hazard_aware", comps(abl["full"]), comps(abl["wo_hazard_aware"])),
    ])

    # ---- Observability family (per visibility level, Holm within family) ----
    obs_files = sorted(RES.glob("observability_final_*.json"))
    if obs_files:
        obs = json.loads(obs_files[-1].read_text())["results"]
        pairs = []
        for o in ["full", "r3", "stale5", "loss50", "priv3"]:
            pairs.append((f"full_daco vs astar [{o}]",
                          comps(obs["full_cms_daco"][o]),
                          comps(obs["astar"][o])))
            if o in obs.get("dstar", {}):
                pairs.append((f"full_daco vs dstar [{o}]",
                              comps(obs["full_cms_daco"][o]),
                              comps(obs["dstar"][o])))
        out["observability"] = compare_family("Observability", pairs)

    # ---- Extreme family ----
    ext_files = sorted(RES.glob("extreme_final_*.json"))
    if ext_files:
        ext = json.loads(ext_files[-1].read_text())["results"]
        out["extreme"] = compare_family("Extreme", [
            ("daco vs astar", comps(ext["full_cms_daco"]), comps(ext["astar"])),
            ("daco vs distance", comps(ext["full_cms_daco"]), comps(ext["distance"])),
            ("distance vs astar", comps(ext["distance"]), comps(ext["astar"])),
            ("astar vs dstar", comps(ext["astar"]), comps(ext["dstar"])),
        ])

    # merge with existing legacy summary
    prev_stats = RES / "stats_summary.json"
    payload = {"paired": out}
    if prev_stats.exists():
        payload["legacy_independent"] = json.loads(prev_stats.read_text())
    (RES / "stats_paired.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {RES/'stats_paired.json'}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
