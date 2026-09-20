"""
STEP 75 — FINAL CA-RDQN MULTI-SEED STATISTICAL ANALYSIS

Purpose:
    Read the saved held-out evaluation results for all ten independent
    CA-RDQN seeds and calculate descriptive multi-seed statistics.

Seeds:
    11, 22, 33, 44, 55, 66, 77, 88, 99, 111

Metrics:
    - Mean evaluation reward
    - Unique evaluation actions
    - NH4 change RMSE
    - SNO change RMSE
    - Relative NH4 improvement
    - Relative SNO improvement

Statistics:
    - Mean
    - Standard deviation
    - Median
    - Minimum
    - Maximum
    - Range
    - 95% confidence interval of the seed-level mean
    - Coefficient of variation
"""

from pathlib import Path
import json
import math
import statistics


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_ca_rdqn"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multiseed_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]


# ================================================================
# STATISTICAL FUNCTIONS
# ================================================================

def mean(values):
    return statistics.mean(values)


def sd(values):
    return statistics.stdev(values)


def median(values):
    return statistics.median(values)


def minimum(values):
    return min(values)


def maximum(values):
    return max(values)


def data_range(values):
    return max(values) - min(values)


def coefficient_of_variation(values):
    m = mean(values)

    if m == 0:
        return float("nan")

    return (sd(values) / abs(m)) * 100.0


def ci95_mean(values):
    """
    95% confidence interval for the mean of the ten independent
    seed-level observations.

    Uses the two-sided t critical value for n-1 degrees of freedom.

    For n = 10:
        df = 9
        t_(0.975,9) = 2.262157...
    """

    n = len(values)

    if n < 2:
        return float("nan"), float("nan")

    m = mean(values)
    s = sd(values)

    t_critical = 2.2621571627

    margin = t_critical * s / math.sqrt(n)

    return m - margin, m + margin


def summarize(values):
    ci_low, ci_high = ci95_mean(values)

    return {
        "n": len(values),
        "mean": mean(values),
        "sd": sd(values),
        "median": median(values),
        "min": minimum(values),
        "max": maximum(values),
        "range": data_range(values),
        "cv_percent": coefficient_of_variation(values),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


# ================================================================
# LOAD RESULTS
# ================================================================

print("=" * 70)
print("STEP 75 — FINAL CA-RDQN MULTI-SEED STATISTICAL ANALYSIS")
print("=" * 70)

print(f"[INFO] Results directory: {RESULTS_DIR}")
print(f"[INFO] Expected seeds: {SEEDS}")
print()


records = []


for seed in SEEDS:

    result_file = (
        RESULTS_DIR
        / f"ca_rdqn_seed_{seed}_results.json"
    )

    if not result_file.exists():
        raise FileNotFoundError(
            f"Missing result file for seed {seed}: "
            f"{result_file}"
        )

    with open(result_file, "r", encoding="utf-8") as f:
        result = json.load(f)

    # ------------------------------------------------------------
    # Verify seed identity
    # ------------------------------------------------------------

    if int(result["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch in {result_file}. "
            f"Expected {seed}, found {result['seed']}."
        )

    # ------------------------------------------------------------
    # Verify common experimental structure
    # ------------------------------------------------------------

    if result["evaluation_steps"] != 478:
        raise ValueError(
            f"Seed {seed}: unexpected evaluation_steps "
            f"{result['evaluation_steps']}; expected 478."
        )

    if result["training_episodes"] != 24:
        raise ValueError(
            f"Seed {seed}: unexpected training_episodes "
            f"{result['training_episodes']}; expected 24."
        )

    if result["training_steps"] != 20616:
        raise ValueError(
            f"Seed {seed}: unexpected training_steps "
            f"{result['training_steps']}; expected 20616."
        )

    records.append(result)

    print(
        f"[PASS] Seed {seed}: "
        f"evaluation={result['evaluation_steps']}, "
        f"NH4 RMSE={result['nh4_rmse']:.6f}, "
        f"SNO RMSE={result['sno_rmse']:.6f}"
    )


print()
print("[PASS] All ten seed result files loaded.")
print("[PASS] Seed identities verified.")
print("[PASS] Common evaluation structure verified.")
print()


# ================================================================
# EXTRACT METRICS
# ================================================================

metrics = {
    "Mean evaluation reward": [
        r["mean_evaluation_reward"]
        for r in records
    ],

    "Unique evaluation actions": [
        r["unique_evaluation_actions"]
        for r in records
    ],

    "NH4 change RMSE": [
        r["nh4_rmse"]
        for r in records
    ],

    "SNO change RMSE": [
        r["sno_rmse"]
        for r in records
    ],

    "NH4 relative improvement (%)": [
        r["nh4_relative_improvement_percent"]
        for r in records
    ],

    "SNO relative improvement (%)": [
        r["sno_relative_improvement_percent"]
        for r in records
    ],
}


# ================================================================
# CALCULATE STATISTICS
# ================================================================

statistics_table = {}

for metric_name, values in metrics.items():

    statistics_table[metric_name] = summarize(values)


# ================================================================
# PRINT SEED-LEVEL RESULTS
# ================================================================

print("=" * 70)
print("SEED-LEVEL HELD-OUT RESULTS")
print("=" * 70)

print(
    f"{'Seed':>6} "
    f"{'Reward':>14} "
    f"{'Actions':>10} "
    f"{'NH4 RMSE':>12} "
    f"{'SNO RMSE':>12} "
    f"{'NH4 Imp.%':>12} "
    f"{'SNO Imp.%':>12}"
)

print("-" * 70)

for r in records:

    print(
        f"{r['seed']:>6} "
        f"{r['mean_evaluation_reward']:>14.6f} "
        f"{r['unique_evaluation_actions']:>10} "
        f"{r['nh4_rmse']:>12.6f} "
        f"{r['sno_rmse']:>12.6f} "
        f"{r['nh4_relative_improvement_percent']:>12.2f} "
        f"{r['sno_relative_improvement_percent']:>12.2f}"
    )


# ================================================================
# PRINT MULTI-SEED STATISTICS
# ================================================================

print()
print("=" * 70)
print("MULTI-SEED DESCRIPTIVE STATISTICS")
print("=" * 70)

for metric_name, stats in statistics_table.items():

    print()
    print(metric_name)
    print("-" * len(metric_name))

    print(f"n       : {stats['n']}")
    print(f"Mean    : {stats['mean']:.6f}")
    print(f"SD      : {stats['sd']:.6f}")
    print(f"Median  : {stats['median']:.6f}")
    print(f"Minimum : {stats['min']:.6f}")
    print(f"Maximum : {stats['max']:.6f}")
    print(f"Range   : {stats['range']:.6f}")
    print(f"CV (%)  : {stats['cv_percent']:.3f}")
    print(
        f"95% CI  : "
        f"[{stats['ci95_low']:.6f}, "
        f"{stats['ci95_high']:.6f}]"
    )


# ================================================================
# VERIFY COMMON BASELINES
# ================================================================

nh4_baselines = [
    r["nh4_zero_change_baseline"]
    for r in records
]

sno_baselines = [
    r["sno_zero_change_baseline"]
    for r in records
]


if max(nh4_baselines) - min(nh4_baselines) > 1e-12:
    raise ValueError(
        "NH4 zero-change baseline differs across seeds."
    )

if max(sno_baselines) - min(sno_baselines) > 1e-12:
    raise ValueError(
        "SNO zero-change baseline differs across seeds."
    )

nh4_baseline = nh4_baselines[0]
sno_baseline = sno_baselines[0]

print()
print("=" * 70)
print("BASELINE CONSISTENCY CHECK")
print("=" * 70)

print(
    f"[PASS] NH4 zero-change baseline: "
    f"{nh4_baseline:.12f}"
)

print(
    f"[PASS] SNO zero-change baseline: "
    f"{sno_baseline:.12f}"
)


# ================================================================
# SAVE JSON SUMMARY
# ================================================================

summary_output = {
    "analysis": {
        "number_of_seeds": len(SEEDS),
        "seeds": SEEDS,
        "evaluation_steps_per_seed": 478,
        "training_episodes_per_seed": 24,
        "training_steps_per_seed": 20616,
    },

    "baselines": {
        "nh4_zero_change_baseline": nh4_baseline,
        "sno_zero_change_baseline": sno_baseline,
    },

    "seed_level_results": records,

    "multi_seed_statistics": statistics_table,
}


json_output = (
    OUTPUT_DIR
    / "ca_rdqn_multiseed_statistics.json"
)


with open(json_output, "w", encoding="utf-8") as f:
    json.dump(
        summary_output,
        f,
        indent=2,
    )


# ================================================================
# SAVE HUMAN-READABLE REPORT
# ================================================================

report_output = (
    OUTPUT_DIR
    / "ca_rdqn_multiseed_statistics.txt"
)


with open(report_output, "w", encoding="utf-8") as f:

    f.write(
        "STEP 75 — FINAL CA-RDQN MULTI-SEED "
        "STATISTICAL ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Seeds: {SEEDS}\n"
        f"Number of independent seeds: {len(SEEDS)}\n"
        f"Evaluation steps per seed: 478\n"
        f"Training episodes per seed: 24\n"
        f"Training steps per seed: 20616\n\n"
    )

    f.write(
        "SEED-LEVEL RESULTS\n"
    )

    f.write("-" * 70 + "\n")

    for r in records:

        f.write(
            f"Seed {r['seed']}: "
            f"Reward={r['mean_evaluation_reward']:.6f}, "
            f"Actions={r['unique_evaluation_actions']}, "
            f"NH4_RMSE={r['nh4_rmse']:.6f}, "
            f"SNO_RMSE={r['sno_rmse']:.6f}, "
            f"NH4_Improvement={r['nh4_relative_improvement_percent']:.4f}%, "
            f"SNO_Improvement={r['sno_relative_improvement_percent']:.4f}%\n"
        )

    f.write("\n")
    f.write(
        "MULTI-SEED DESCRIPTIVE STATISTICS\n"
    )

    f.write("-" * 70 + "\n")

    for metric_name, stats in statistics_table.items():

        f.write(f"\n{metric_name}\n")
        f.write(f"  n       = {stats['n']}\n")
        f.write(f"  mean    = {stats['mean']:.6f}\n")
        f.write(f"  SD      = {stats['sd']:.6f}\n")
        f.write(f"  median  = {stats['median']:.6f}\n")
        f.write(f"  minimum = {stats['min']:.6f}\n")
        f.write(f"  maximum = {stats['max']:.6f}\n")
        f.write(f"  range   = {stats['range']:.6f}\n")
        f.write(f"  CV (%)  = {stats['cv_percent']:.3f}\n")
        f.write(
            f"  95% CI  = "
            f"[{stats['ci95_low']:.6f}, "
            f"{stats['ci95_high']:.6f}]\n"
        )

    f.write("\n")
    f.write("BASELINES\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"NH4 zero-change baseline = "
        f"{nh4_baseline:.12f}\n"
    )
    f.write(
        f"SNO zero-change baseline = "
        f"{sno_baseline:.12f}\n"
    )


# ================================================================
# FINAL STATUS
# ================================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(f"[PASS] JSON summary:   {json_output}")
print(f"[PASS] Text report:     {report_output}")

print()
print("=" * 70)
print("STEP 75 COMPLETED")
print("=" * 70)