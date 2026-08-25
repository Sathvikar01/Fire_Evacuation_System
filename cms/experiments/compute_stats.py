"""Compute Welch t-tests and Cohen's d for key comparisons; emit stats JSON."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).parent / "results"


def load_final(prefix):
    files = sorted(RES.glob(f"{prefix}_final_*.json"))
    return json.loads(files[-1].read_text())


def welch_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if sp == 0:
        return 0.0
    return float((a.mean()-b.mean()) / sp)


def compare(name, a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = welch_d(a, b)
    print(f"{name:<42} meanA={np.mean(a):.4f} meanB={np.mean(b):.4f} "
          f"d={d:+.2f} t={t:+.2f} p={p:.4g}")
    return {"comparison": name, "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)), "cohens_d": d,
            "welch_t": float(t), "p_value": float(p)}


def comps(entry):
    return [r["completion_rate"] for r in entry]


def main():
    out = {}
    base = load_final("baselines")["results"]
    hard = load_final("hard_conditions_final".replace("_final","")) if False else None
    hard_files = sorted(RES.glob("hard_conditions_final_*.json"))
    hard = json.loads(hard_files[-1].read_text())["results"]
    abl = load_final("ablation")["results"]

    print("=== EASY: full vs others (completion) ===")
    out["easy"] = [
        compare("full vs astar", comps(base["full_cms_daco"]), comps(base["astar"])),
        compare("full vs distance", comps(base["full_cms_daco"]), comps(base["distance"])),
        compare("full vs standard_aco", comps(base["full_cms_daco"]), comps(base["standard_aco"])),
        compare("astar vs random", comps(base["astar"]), [r["completion_rate"] for r in json.loads((RES/"base_random_1_30.json").read_text())["results"]]),
    ]
    print("\n=== HARD: full vs others (completion) ===")
    rnd_hard = [r["completion_rate"] for r in json.loads((RES/"hard_random_1_30.json").read_text())["results"]]
    out["hard"] = [
        compare("full vs astar", comps(hard["full_cms_daco"]), comps(hard["astar"])),
        compare("full vs distance", comps(hard["full_cms_daco"]), comps(hard["distance"])),
        compare("full vs standard_aco", comps(hard["full_cms_daco"]), comps(hard["standard_aco"])),
        compare("astar vs random", comps(hard["astar"]), rnd_hard),
        compare("distance vs random", comps(hard["distance"]), rnd_hard),
    ]
    print("\n=== ABLATION (completion) ===")
    out["ablation"] = [
        compare("full vs wo_bfs_seed", comps(abl["full"]), comps(abl["wo_bfs_seed"])),
        compare("full vs wo_dual", comps(abl["full"]), comps(abl["wo_dual"])),
        compare("full vs wo_predictive", comps(abl["full"]), comps(abl["wo_predictive"])),
        compare("full vs wo_escape", comps(abl["full"]), comps(abl["wo_escape"])),
        compare("full vs wo_hazard_aware", comps(abl["full"]), comps(abl["wo_hazard_aware"])),
    ]

    rob_files = sorted(RES.glob("robustness_final_*.json"))
    rob = json.loads(rob_files[-1].read_text())["summary"]
    print("\n=== ROBUSTNESS walls slope check ===")
    w = {k: v["completion_mean"] for k, v in rob["walls"].items()}
    print(w)

    path = RES / "stats_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
