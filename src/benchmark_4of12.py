from pathlib import Path
import itertools
import numpy as np

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_FILE = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

TRAIN_END = 863
LEAD = 2


SENSOR_NAMES = [
    "DO_R1",
    "DO_R2",
    "DO_R3",
    "DO_R4",
    "DO_R5",
    "NH4_R1",
    "NH4_R2",
    "NH4_R3",
    "NH4_R4",
    "NH4_R5",
    "SNO_R3",
    "SNO_R5",
]


# ------------------------------------------------------------
# Run BSM1
# ------------------------------------------------------------

print("Running BSM1 simulation...")

data = np.loadtxt(DATA_FILE, delimiter=",")

model = BSM1OL(
    data_in=data,
    timestep=None,
)

model.simulate(plot=False)

print("BSM1 simulation complete.")


# ------------------------------------------------------------
# Construct the 12-sensor candidate matrix
# ------------------------------------------------------------

X = np.column_stack(
    [
        model.y_out1_all[:, 7],
        model.y_out2_all[:, 7],
        model.y_out3_all[:, 7],
        model.y_out4_all[:, 7],
        model.y_out5_all[:, 7],
        model.y_out1_all[:, 9],
        model.y_out2_all[:, 9],
        model.y_out3_all[:, 9],
        model.y_out4_all[:, 9],
        model.y_out5_all[:, 9],
        model.y_out3_all[:, 8],
        model.y_out5_all[:, 8],
    ]
)


# ------------------------------------------------------------
# Construct 30-minute-ahead change targets
#
# 15-minute simulation interval
# LEAD = 2 therefore represents 30 minutes
# ------------------------------------------------------------

Y = np.column_stack(
    [
        model.ys_eff_all[:, 9],
        model.ys_eff_all[:, 8],
    ]
)

end = len(X) - LEAD

X_forecast = X[:end]

delta_Y = Y[LEAD:] - Y[:end]


# ------------------------------------------------------------
# Chronological train/evaluation split
# ------------------------------------------------------------

X_train = X_forecast[:TRAIN_END]
Y_train = delta_Y[:TRAIN_END]

X_eval = X_forecast[TRAIN_END:]
Y_eval = delta_Y[TRAIN_END:]


print()
print("Benchmark configuration")
print("-----------------------")
print(f"Total observations: {len(X)}")
print(f"Training observations: {len(X_train)}")
print(f"Evaluation observations: {len(X_eval)}")
print(f"Forecast horizon: {LEAD * 15} minutes")
print(f"Candidate sensors: {X.shape[1]}")
print()


# ------------------------------------------------------------
# RMSE function
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


# ------------------------------------------------------------
# Full 12-sensor reference
# ------------------------------------------------------------

A12 = np.column_stack(
    [
        np.ones(len(X_train)),
        X_train,
    ]
)

B12 = np.linalg.lstsq(
    A12,
    Y_train,
    rcond=None,
)[0]

P12 = np.column_stack(
    [
        np.ones(len(X_eval)),
        X_eval,
    ]
) @ B12

rmse_12 = rmse(
    Y_eval,
    P12,
)


# ------------------------------------------------------------
# Zero-change baseline
# ------------------------------------------------------------

zero_prediction = np.zeros_like(Y_eval)

rmse_zero = rmse(
    Y_eval,
    zero_prediction,
)


# ------------------------------------------------------------
# Exhaustive 4-of-12 evaluation
# ------------------------------------------------------------

combinations = list(
    itertools.combinations(
        range(12),
        4,
    )
)

results = []

print("Evaluating all 495 four-sensor combinations...")

for combination in combinations:

    Xtr = X_train[:, combination]
    Xev = X_eval[:, combination]

    A = np.column_stack(
        [
            np.ones(len(Xtr)),
            Xtr,
        ]
    )

    coefficients = np.linalg.lstsq(
        A,
        Y_train,
        rcond=None,
    )[0]

    prediction = np.column_stack(
        [
            np.ones(len(Xev)),
            Xev,
        ]
    ) @ coefficients

    errors = rmse(
        Y_eval,
        prediction,
    )

    results.append(
        {
            "combination": combination,
            "nh4_rmse": errors[0],
            "sno_rmse": errors[1],
        }
    )


# ------------------------------------------------------------
# Extract results
# ------------------------------------------------------------

nh4_values = np.array(
    [
        result["nh4_rmse"]
        for result in results
    ]
)

sno_values = np.array(
    [
        result["sno_rmse"]
        for result in results
    ]
)


best_nh4_index = np.argmin(nh4_values)
best_sno_index = np.argmin(sno_values)

worst_nh4_index = np.argmax(nh4_values)
worst_sno_index = np.argmax(sno_values)


best_nh4 = results[best_nh4_index]
best_sno = results[best_sno_index]

worst_nh4 = results[worst_nh4_index]
worst_sno = results[worst_sno_index]


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

def sensor_names(combination):
    return ", ".join(
        SENSOR_NAMES[i]
        for i in combination
    )


print()
print("=" * 60)
print("EXHAUSTIVE 4-OF-12 BENCHMARK")
print("=" * 60)

print()
print("REFERENCE RESULTS")
print("-----------------")

print(
    f"Full-12 NH4 RMSE: "
    f"{rmse_12[0]:.6f}"
)

print(
    f"Full-12 SNO RMSE: "
    f"{rmse_12[1]:.6f}"
)

print(
    f"Zero-change NH4 RMSE: "
    f"{rmse_zero[0]:.6f}"
)

print(
    f"Zero-change SNO RMSE: "
    f"{rmse_zero[1]:.6f}"
)


print()
print("FOUR-SENSOR DISTRIBUTION")
print("------------------------")

print(
    f"NH4 mean: "
    f"{nh4_values.mean():.6f}"
)

print(
    f"NH4 median: "
    f"{np.median(nh4_values):.6f}"
)

print(
    f"NH4 std: "
    f"{nh4_values.std():.6f}"
)

print(
    f"NH4 best: "
    f"{nh4_values.min():.6f}"
)

print(
    f"NH4 worst: "
    f"{nh4_values.max():.6f}"
)

print()

print(
    f"SNO mean: "
    f"{sno_values.mean():.6f}"
)

print(
    f"SNO median: "
    f"{np.median(sno_values):.6f}"
)

print(
    f"SNO std: "
    f"{sno_values.std():.6f}"
)

print(
    f"SNO best: "
    f"{sno_values.min():.6f}"
)

print(
    f"SNO worst: "
    f"{sno_values.max():.6f}"
)


print()
print("BEST FOUR-SENSOR CONFIGURATIONS")
print("--------------------------------")

print(
    "Best NH4 sensors:",
    sensor_names(best_nh4["combination"]),
)

print(
    f"Best NH4 RMSE: "
    f"{best_nh4['nh4_rmse']:.6f}"
)

print()

print(
    "Best SNO sensors:",
    sensor_names(best_sno["combination"]),
)

print(
    f"Best SNO RMSE: "
    f"{best_sno['sno_rmse']:.6f}"
)


print()
print("WORST FOUR-SENSOR CONFIGURATIONS")
print("---------------------------------")

print(
    "Worst NH4 sensors:",
    sensor_names(worst_nh4["combination"]),
)

print(
    f"Worst NH4 RMSE: "
    f"{worst_nh4['nh4_rmse']:.6f}"
)

print()

print(
    "Worst SNO sensors:",
    sensor_names(worst_sno["combination"]),
)

print(
    f"Worst SNO RMSE: "
    f"{worst_sno['sno_rmse']:.6f}"
)

# ------------------------------------------------------------
# Save exhaustive 495-configuration results
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "benchmark_4of12_results.csv"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write(
        "action_index,"
        "sensor_1,"
        "sensor_2,"
        "sensor_3,"
        "sensor_4,"
        "nh4_rmse,"
        "sno_rmse\n"
    )

    for action_index, result in enumerate(results):

        combination = result["combination"]

        f.write(
            f"{action_index},"
            f"{SENSOR_NAMES[combination[0]]},"
            f"{SENSOR_NAMES[combination[1]]},"
            f"{SENSOR_NAMES[combination[2]]},"
            f"{SENSOR_NAMES[combination[3]]},"
            f"{result['nh4_rmse']:.10f},"
            f"{result['sno_rmse']:.10f}\n"
        )

print()
print(f"[SAVED] {OUTPUT_FILE}")


print()
print("=" * 60)
print("Benchmark complete.")
print("=" * 60)


