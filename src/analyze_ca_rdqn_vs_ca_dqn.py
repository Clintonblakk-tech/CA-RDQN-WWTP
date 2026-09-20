"""
STEP 77 — PAIRED CA-RDQN vs CA-DQN MULTI-SEED ABLATION ANALYSIS

Compares the ten final independent seeds:
11, 22, 33, 44, 55, 66, 77, 88, 99, 111

CA-RDQN:
    data/processed/final_ca_rdqn/

CA-DQN:
    data/processed/final_ca_dqn_ablation/

No model retraining is performed.
All statistics are calculated from the saved seed-level JSON results.
"""

from pathlib import Path
import json
import math
import statistics

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

ROOT = Path(__file__).resolve().parents[1]

RDQN_DIR = ROOT / "data" / "processed" / "final_ca_rdqn"
DQN_DIR = ROOT / "data" / "processed" / "final_ca_dqn_ablation"

OUT_DIR = ROOT / "data" / "processed" / "ca_rdqn_vs_ca_dqn_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(data, candidates):
    """
    Search recursively for a metric key.
    """
    if isinstance(data, dict):
        for key in candidates:
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    return float(value)

        for value in data.values():
            result = get_metric(value, candidates)
            if result is not None:
                return result

    elif isinstance(data, list):
        for value in data:
            result = get_metric(value, candidates)
            if result is not None:
                return result

    return None


def load_seed_results(directory, prefix, seed):
    path = directory / f"{prefix}_{seed}_results.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    data = load_json(path)

    nh4 = get_metric(
        data,
        [
            "eval_nh4_rmse",
            "nh4_rmse",
            "delta_nh4_rmse",
            "eval_delta_nh4_rmse",
            "NH4_RMSE",
        ],
    )

    sno = get_metric(
        data,
        [
            "eval_sno_rmse",
            "sno_rmse",
            "delta_sno_rmse",
            "eval_delta_sno_rmse",
            "SNO_RMSE",
        ],
    )

    reward = get_metric(
        data,
        [
            "mean_eval_reward",
            "eval_mean_reward",
            "mean_reward",
            "evaluation_mean_reward",
        ],
    )

    unique_actions = get_metric(
        data,
        [
            "unique_eval_actions",
            "eval_unique_actions",
            "unique_actions",
            "evaluation_unique_actions",
        ],
    )

    if nh4 is None or sno is None:
        raise ValueError(
            f"Could not identify NH4/SNO RMSE in {path}. "
            f"Inspect this JSON before proceeding."
        )

    return {
        "seed": seed,
        "nh4_rmse": nh4,
        "sno_rmse": sno,
        "mean_reward": reward,
        "unique_actions": unique_actions,
        "file": str(path),
    }


def mean(values):
    return statistics.mean(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def median(values):
    return statistics.median(values)


def ci95(values):
    """
    Approximate 95% CI using t critical value for n=10.
    t_0.975,9 = 2.262157.
    """
    n = len(values)

    if n < 2:
        return [None, None]

    m = mean(values)
    se = sd(values) / math.sqrt(n)
    tcrit = 2.2621571627

    return [
        m - tcrit * se,
        m + tcrit * se,
    ]


def paired_summary(rdqn_values, dqn_values):
    differences = [
        dqn - rdqn
        for rdqn, dqn in zip(rdqn_values, dqn_values)
    ]

    return {
        "differences_dqn_minus_rdqn": differences,
        "mean_difference": mean(differences),
        "sd_difference": sd(differences),
        "median_difference": median(differences),
        "min_difference": min(differences),
        "max_difference": max(differences),
        "ci95_difference": ci95(differences),
    }


def try_scipy_tests(rdqn_values, dqn_values):
    """
    Perform paired t-test and Wilcoxon signed-rank test if SciPy
    is available.
    """
    result = {}

    try:
        from scipy.stats import ttest_rel, wilcoxon

        ttest = ttest_rel(dqn_values, rdqn_values)

        result["paired_t_test"] = {
            "statistic": float(ttest.statistic),
            "p_value": float(ttest.pvalue),
        }

        try:
            wx = wilcoxon(
                dqn_values,
                rdqn_values,
                alternative="two-sided",
                zero_method="wilcox",
            )

            result["wilcoxon_signed_rank"] = {
                "statistic": float(wx.statistic),
                "p_value": float(wx.pvalue),
            }

        except ValueError as exc:
            result["wilcoxon_signed_rank"] = {
                "statistic": None,
                "p_value": None,
                "note": str(exc),
            }

    except ImportError:
        result["statistics_package"] = (
            "SciPy not available; inferential tests were not calculated."
        )

    return result


def summarize(values):
    return {
        "mean": mean(values),
        "sd": sd(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "ci95": ci95(values),
    }


print("=" * 78)
print("STEP 77 — PAIRED CA-RDQN vs CA-DQN MULTI-SEED ABLATION ANALYSIS")
print("=" * 78)

rdqn = {}
dqn = {}

for seed in SEEDS:
    rdqn[seed] = load_seed_results(
        RDQN_DIR,
        "ca_rdqn_seed",
        seed,
    )

    dqn[seed] = load_seed_results(
        DQN_DIR,
        "ca_dqn_ablation_seed",
        seed,
    )

print("\n[PASS] All ten CA-RDQN result files loaded.")
print("[PASS] All ten CA-DQN result files loaded.")
print("[PASS] Seed pairing verified.")

# ------------------------------------------------------------
# Extract metrics
# ------------------------------------------------------------

rdqn_nh4 = [rdqn[s]["nh4_rmse"] for s in SEEDS]
dqn_nh4 = [dqn[s]["nh4_rmse"] for s in SEEDS]

rdqn_sno = [rdqn[s]["sno_rmse"] for s in SEEDS]
dqn_sno = [dqn[s]["sno_rmse"] for s in SEEDS]

rdqn_reward = [
    rdqn[s]["mean_reward"]
    for s in SEEDS
    if rdqn[s]["mean_reward"] is not None
]

dqn_reward = [
    dqn[s]["mean_reward"]
    for s in SEEDS
    if dqn[s]["mean_reward"] is not None
]

rdqn_unique = [
    rdqn[s]["unique_actions"]
    for s in SEEDS
    if rdqn[s]["unique_actions"] is not None
]

dqn_unique = [
    dqn[s]["unique_actions"]
    for s in SEEDS
    if dqn[s]["unique_actions"] is not None
]

# ------------------------------------------------------------
# Paired differences
# ------------------------------------------------------------

nh4_paired = paired_summary(rdqn_nh4, dqn_nh4)
sno_paired = paired_summary(rdqn_sno, dqn_sno)

nh4_tests = try_scipy_tests(rdqn_nh4, dqn_nh4)
sno_tests = try_scipy_tests(rdqn_sno, dqn_sno)

# ------------------------------------------------------------
# Seed-level table
# ------------------------------------------------------------

seed_table = []

for i, seed in enumerate(SEEDS):
    seed_table.append(
        {
            "seed": seed,
            "rdqn_nh4_rmse": rdqn_nh4[i],
            "dqn_nh4_rmse": dqn_nh4[i],
            "dqn_minus_rdqn_nh4": dqn_nh4[i] - rdqn_nh4[i],
            "rdqn_sno_rmse": rdqn_sno[i],
            "dqn_sno_rmse": dqn_sno[i],
            "dqn_minus_rdqn_sno": dqn_sno[i] - rdqn_sno[i],
            "rdqn_unique_actions": rdqn[seed]["unique_actions"],
            "dqn_unique_actions": dqn[seed]["unique_actions"],
        }
    )

# ------------------------------------------------------------
# Overall results
# ------------------------------------------------------------

results = {
    "analysis": {
        "step": 77,
        "description": (
            "Paired ten-seed comparison of CA-RDQN and CA-DQN "
            "under the locked experimental protocol."
        ),
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
    },

    "ca_rdqn": {
        "nh4_rmse": summarize(rdqn_nh4),
        "sno_rmse": summarize(rdqn_sno),
    },

    "ca_dqn": {
        "nh4_rmse": summarize(dqn_nh4),
        "sno_rmse": summarize(dqn_sno),
    },

    "paired_comparison": {
        "nh4_rmse": {
            **nh4_paired,
            **nh4_tests,
        },
        "sno_rmse": {
            **sno_paired,
            **sno_tests,
        },
    },

    "policy_diversity": {
        "ca_rdqn_unique_actions": (
            summarize(rdqn_unique) if rdqn_unique else None
        ),
        "ca_dqn_unique_actions": (
            summarize(dqn_unique) if dqn_unique else None
        ),
    },

    "seed_level": seed_table,
}

# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

json_path = OUT_DIR / "ca_rdqn_vs_ca_dqn_paired_statistics.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# ------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------

txt_path = OUT_DIR / "ca_rdqn_vs_ca_dqn_paired_statistics.txt"

with open(txt_path, "w", encoding="utf-8") as f:

    f.write("STEP 77 — PAIRED CA-RDQN vs CA-DQN MULTI-SEED ABLATION\n")
    f.write("=" * 78 + "\n\n")

    f.write(
        "Ten independent seeds: "
        + ", ".join(map(str, SEEDS))
        + "\n\n"
    )

    f.write("CA-RDQN\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Delta NH4 RMSE: "
        f"{mean(rdqn_nh4):.6f} ± {sd(rdqn_nh4):.6f}\n"
    )
    f.write(
        f"Delta SNO RMSE: "
        f"{mean(rdqn_sno):.6f} ± {sd(rdqn_sno):.6f}\n"
    )

    if rdqn_unique:
        f.write(
            f"Unique evaluation actions: "
            f"{mean(rdqn_unique):.3f} ± {sd(rdqn_unique):.3f}\n"
        )

    f.write("\nCA-DQN\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Delta NH4 RMSE: "
        f"{mean(dqn_nh4):.6f} ± {sd(dqn_nh4):.6f}\n"
    )
    f.write(
        f"Delta SNO RMSE: "
        f"{mean(dqn_sno):.6f} ± {sd(dqn_sno):.6f}\n"
    )

    if dqn_unique:
        f.write(
            f"Unique evaluation actions: "
            f"{mean(dqn_unique):.3f} ± {sd(dqn_unique):.3f}\n"
        )

    f.write("\nPAIRED DIFFERENCES: CA-DQN − CA-RDQN\n")
    f.write("-" * 40 + "\n")

    f.write(
        f"Delta NH4 RMSE difference: "
        f"{nh4_paired['mean_difference']:.6f} ± "
        f"{nh4_paired['sd_difference']:.6f}\n"
    )

    f.write(
        f"95% CI: "
        f"[{nh4_paired['ci95_difference'][0]:.6f}, "
        f"{nh4_paired['ci95_difference'][1]:.6f}]\n"
    )

    f.write(
        f"Delta SNO RMSE difference: "
        f"{sno_paired['mean_difference']:.6f} ± "
        f"{sno_paired['sd_difference']:.6f}\n"
    )

    f.write(
        f"95% CI: "
        f"[{sno_paired['ci95_difference'][0]:.6f}, "
        f"{sno_paired['ci95_difference'][1]:.6f}]\n"
    )

    f.write("\nINFERENTIAL TESTS\n")
    f.write("-" * 40 + "\n")

    if "paired_t_test" in nh4_tests:
        f.write(
            f"NH4 paired t-test: "
            f"t = {nh4_tests['paired_t_test']['statistic']:.6f}, "
            f"p = {nh4_tests['paired_t_test']['p_value']:.6f}\n"
        )

    if "wilcoxon_signed_rank" in nh4_tests:
        wx = nh4_tests["wilcoxon_signed_rank"]
        if wx["p_value"] is not None:
            f.write(
                f"NH4 Wilcoxon: "
                f"W = {wx['statistic']:.6f}, "
                f"p = {wx['p_value']:.6f}\n"
            )

    if "paired_t_test" in sno_tests:
        f.write(
            f"SNO paired t-test: "
            f"t = {sno_tests['paired_t_test']['statistic']:.6f}, "
            f"p = {sno_tests['paired_t_test']['p_value']:.6f}\n"
        )

    if "wilcoxon_signed_rank" in sno_tests:
        wx = sno_tests["wilcoxon_signed_rank"]
        if wx["p_value"] is not None:
            f.write(
                f"SNO Wilcoxon: "
                f"W = {wx['statistic']:.6f}, "
                f"p = {wx['p_value']:.6f}\n"
            )

    f.write("\nSEED-LEVEL RESULTS\n")
    f.write("-" * 78 + "\n")

    f.write(
        "Seed | RDQN NH4 | DQN NH4 | DQN-RDQN | "
        "RDQN SNO | DQN SNO | DQN-RDQN\n"
    )

    for row in seed_table:
        f.write(
            f"{row['seed']:>4} | "
            f"{row['rdqn_nh4_rmse']:.6f} | "
            f"{row['dqn_nh4_rmse']:.6f} | "
            f"{row['dqn_minus_rdqn_nh4']:+.6f} | "
            f"{row['rdqn_sno_rmse']:.6f} | "
            f"{row['dqn_sno_rmse']:.6f} | "
            f"{row['dqn_minus_rdqn_sno']:+.6f}\n"
        )

print("\n" + "=" * 78)
print("STEP 77 COMPLETE")
print("=" * 78)

print(f"\nOutput directory:")
print(OUT_DIR)

print(f"\nJSON:")
print(json_path)

print(f"\nText report:")
print(txt_path)

print("\nCA-RDQN:")
print(
    f"  NH4 RMSE = {mean(rdqn_nh4):.6f} ± {sd(rdqn_nh4):.6f}"
)
print(
    f"  SNO RMSE = {mean(rdqn_sno):.6f} ± {sd(rdqn_sno):.6f}"
)

print("\nCA-DQN:")
print(
    f"  NH4 RMSE = {mean(dqn_nh4):.6f} ± {sd(dqn_nh4):.6f}"
)
print(
    f"  SNO RMSE = {mean(dqn_sno):.6f} ± {sd(dqn_sno):.6f}"
)

print("\nPaired difference — DQN minus RDQN:")

print(
    f"  NH4 = {nh4_paired['mean_difference']:.6f}"
)

print(
    f"  SNO = {sno_paired['mean_difference']:.6f}"
)

print("\n[PASS] Step 77 analysis completed.")