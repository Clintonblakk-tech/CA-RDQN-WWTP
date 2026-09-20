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
WINDOW = 96  # 96 observations = 24 hours at 15-minute intervals


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
# RMSE
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2,
            axis=0,
        )
    )


# ------------------------------------------------------------
# Run BSM1
# ------------------------------------------------------------

print("Running BSM1 simulation...")

data = np.loadtxt(
    DATA_FILE,
    delimiter=",",
)

model = BSM1OL(
    data_in=data,
    timestep=None,
)

model.simulate(
    plot=False,
)

print("BSM1 simulation complete.")


# ------------------------------------------------------------
# Candidate sensor matrix
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
# Effluent targets
# ------------------------------------------------------------

Y = np.column_stack(
    [
        model.ys_eff_all[:, 9],
        model.ys_eff_all[:, 8],
    ]
)


# ------------------------------------------------------------
# 30-minute change target
# ------------------------------------------------------------

end = len(X) - LEAD

X_forecast = X[:end]

delta_Y = Y[LEAD:] - Y[:end]


# ------------------------------------------------------------
# Chronological training/evaluation split
# ------------------------------------------------------------

X_train = X_forecast[:TRAIN_END]
D_train = delta_Y[:TRAIN_END]

X_eval = X_forecast[TRAIN_END:]
D_eval = delta_Y[TRAIN_END:]


# ------------------------------------------------------------
# Generate all feasible four-sensor actions
# ------------------------------------------------------------

combinations = list(
    itertools.combinations(
        range(12),
        4,
    )
)

print()
print("Temporal oracle configuration")
print("-----------------------------")
print(f"Candidate sensors: {len(SENSOR_NAMES)}")
print(f"Four-sensor combinations: {len(combinations)}")
print(f"Training observations: {len(X_train)}")
print(f"Evaluation observations: {len(X_eval)}")
print(f"Evaluation window: {WINDOW} observations")
print(f"Forecast horizon: {LEAD * 15} minutes")
print()


# ------------------------------------------------------------
# Train one fixed estimator for every four-sensor action
#
# IMPORTANT:
# Coefficients are fitted ONLY on the training period.
# No evaluation observations are used for fitting.
# ------------------------------------------------------------

print("Fitting 495 four-sensor estimators...")

models = []

for combination in combinations:

    Xtr = X_train[:, combination]

    A = np.column_stack(
        [
            np.ones(len(Xtr)),
            Xtr,
        ]
    )

    coefficients = np.linalg.lstsq(
        A,
        D_train,
        rcond=None,
    )[0]

    models.append(
        (
            combination,
            coefficients,
        )
    )


print("Estimator fitting complete.")
print()


# ------------------------------------------------------------
# Divide held-out period into daily windows
# ------------------------------------------------------------

n_eval = len(X_eval)

window_starts = list(
    range(
        0,
        n_eval,
        WINDOW,
    )
)


# ------------------------------------------------------------
# Evaluate every fixed four-sensor estimator in every window
# ------------------------------------------------------------

daily_results = []

for day_number, start in enumerate(
    window_starts,
    start=1,
):

    stop = min(
        start + WINDOW,
        n_eval,
    )

    X_day = X_eval[start:stop]
    D_day = D_eval[start:stop]

    window_results = []

    for combination, coefficients in models:

        X_day_selected = X_day[:, combination]

        prediction = np.column_stack(
            [
                np.ones(len(X_day_selected)),
                X_day_selected,
            ]
        ) @ coefficients

        errors = rmse(
            D_day,
            prediction,
        )

        window_results.append(
            {
                "combination": combination,
                "nh4_rmse": errors[0],
                "sno_rmse": errors[1],
            }
        )

    nh4_values = np.array(
        [
            result["nh4_rmse"]
            for result in window_results
        ]
    )

    sno_values = np.array(
        [
            result["sno_rmse"]
            for result in window_results
        ]
    )

    best_nh4_index = np.argmin(
        nh4_values
    )

    best_sno_index = np.argmin(
        sno_values
    )

    best_nh4 = window_results[
        best_nh4_index
    ]

    best_sno = window_results[
        best_sno_index
    ]

    daily_results.append(
        {
            "day": day_number,
            "observations": stop - start,
            "best_nh4": best_nh4,
            "best_sno": best_sno,
        }
    )


# ------------------------------------------------------------
# Report daily best configurations
# ------------------------------------------------------------

print("=" * 70)
print("TEMPORAL SENSOR ORACLE")
print("=" * 70)

print()

for result in daily_results:

    day = result["day"]

    nh4 = result["best_nh4"]
    sno = result["best_sno"]

    nh4_names = ", ".join(
        SENSOR_NAMES[i]
        for i in nh4["combination"]
    )

    sno_names = ", ".join(
        SENSOR_NAMES[i]
        for i in sno["combination"]
    )

    print(
        f"DAY {day} "
        f"({result['observations']} observations)"
    )

    print(
        f"  Best NH4 configuration: "
        f"{nh4_names}"
    )

    print(
        f"  NH4 RMSE: "
        f"{nh4['nh4_rmse']:.6f}"
    )

    print(
        f"  Best SNO configuration: "
        f"{sno_names}"
    )

    print(
        f"  SNO RMSE: "
        f"{sno['sno_rmse']:.6f}"
    )

    print()


# ------------------------------------------------------------
# Count configuration changes
# ------------------------------------------------------------

nh4_configurations = [
    result["best_nh4"]["combination"]
    for result in daily_results
]

sno_configurations = [
    result["best_sno"]["combination"]
    for result in daily_results
]


unique_nh4 = list(
    dict.fromkeys(
        nh4_configurations
    )
)

unique_sno = list(
    dict.fromkeys(
        sno_configurations
    )
)


print("=" * 70)
print("TEMPORAL VARIABILITY")
print("=" * 70)

print()

print(
    "Unique best NH4 configurations:",
    len(unique_nh4),
)

print(
    "Unique best SNO configurations:",
    len(unique_sno),
)

print()

print(
    "NH4 configuration sequence:"
)

for i, combination in enumerate(
    nh4_configurations,
    start=1,
):

    print(
        f"  Day {i}: "
        f"{', '.join(SENSOR_NAMES[j] for j in combination)}"
    )


print()

print(
    "SNO configuration sequence:"
)

for i, combination in enumerate(
    sno_configurations,
    start=1,
):

    print(
        f"  Day {i}: "
        f"{', '.join(SENSOR_NAMES[j] for j in combination)}"
    )


print()
print("=" * 70)
print("Temporal oracle analysis complete.")
print("=" * 70)

# ------------------------------------------------------------
# Save temporal oracle results
# ------------------------------------------------------------

import json
from pathlib import Path

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "temporal_sensor_oracle"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

temporal_results = []

for result in daily_results:

    temporal_results.append(
        {
            "day": int(result["day"]),
            "observations": int(result["observations"]),

            "best_nh4_configuration": [
                SENSOR_NAMES[i]
                for i in result["best_nh4"]["combination"]
            ],

            "nh4_rmse": float(
                result["best_nh4"]["nh4_rmse"]
            ),

            "best_sno_configuration": [
                SENSOR_NAMES[i]
                for i in result["best_sno"]["combination"]
            ],

            "sno_rmse": float(
                result["best_sno"]["sno_rmse"]
            ),
        }
    )

with open(
    OUTPUT_DIR / "temporal_oracle_results.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        temporal_results,
        f,
        indent=2,
    )

print()
print(
    "Saved temporal oracle results to:"
)
print(
    OUTPUT_DIR
    / "temporal_oracle_results.json"
)