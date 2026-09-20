import itertools
import numpy as np

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

TRAIN_END = 863
LEAD = 2
HISTORY = 4


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


# ============================================================
# FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(
        np.mean((y_true - y_pred) ** 2, axis=0)
    )


def build_history(X, history):
    """
    Construct a flattened historical feature matrix.

    For each prediction time t, the estimator receives:

        [X(t-history), ..., X(t-1)]

    No current target value is included.
    """

    n = len(X)

    rows = []
    for t in range(history, n):
        row = X[t-history:t].reshape(-1)
        rows.append(row)

    return np.asarray(rows)


# ============================================================
# RUN BSM1
# ============================================================

print("Running BSM1 simulation...")

data = np.loadtxt(
    DATA_FILE,
    delimiter=","
)

model = BSM1OL(
    data_in=data,
    timestep=None
)

model.simulate(plot=False)

print("BSM1 simulation complete.")


# ============================================================
# BUILD 12-SENSOR MATRIX
# ============================================================

X = np.column_stack([
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
])


# ============================================================
# TARGET: 30-MINUTE CHANGE
# ============================================================

Y = np.column_stack([
    model.ys_eff_all[:, 9],   # Effluent NH4-N
    model.ys_eff_all[:, 8],   # Effluent SNO
])

end = len(X) - LEAD

X = X[:end]

delta_Y = Y[LEAD:] - Y[:end]


# ============================================================
# CREATE HISTORICAL FEATURES
# ============================================================

X_history = build_history(
    X,
    HISTORY
)

D = delta_Y[HISTORY:]


# ============================================================
# TRAIN / EVALUATION SPLIT
# ============================================================

X_train = X_history[:TRAIN_END - HISTORY]
D_train = D[:TRAIN_END - HISTORY]

X_eval = X_history[TRAIN_END - HISTORY:]
D_eval = D[TRAIN_END - HISTORY:]


print()
print("Estimator interface")
print("-------------------")
print("Candidate sensors:", len(SENSOR_NAMES))
print("Four-sensor combinations:", 495)
print("History length:", HISTORY, "observations")
print("History duration: 1 hour")
print("Forecast horizon: 30 minutes")
print("Training observations:", len(X_train))
print("Evaluation observations:", len(X_eval))


# ============================================================
# EXHAUSTIVE FOUR-SENSOR TEST
# ============================================================

combinations = list(
    itertools.combinations(
        range(12),
        4
    )
)

results = []


print()
print("Testing 495 four-sensor historical estimators...")


for combination in combinations:

    selected_names = [
        SENSOR_NAMES[i]
        for i in combination
    ]

    columns = []

    for sensor_index in combination:

        start = sensor_index
        stop = sensor_index + 1

        # Extract this sensor's four historical observations.
        #
        # History matrix ordering:
        # [sensor1_t-4 ... sensor12_t-4,
        #  sensor1_t-3 ... sensor12_t-3,
        #  sensor1_t-2 ... sensor12_t-2,
        #  sensor1_t-1 ... sensor12_t-1]

        for h in range(HISTORY):

            column = h * 12 + sensor_index
            columns.append(column)

    Xtr = X_train[:, columns]
    Xev = X_eval[:, columns]

    A_train = np.column_stack([
        np.ones(len(Xtr)),
        Xtr
    ])

    coefficients = np.linalg.lstsq(
        A_train,
        D_train,
        rcond=None
    )[0]

    A_eval = np.column_stack([
        np.ones(len(Xev)),
        Xev
    ])

    prediction = A_eval @ coefficients

    errors = rmse(
        D_eval,
        prediction
    )

    results.append({
        "combination": combination,
        "names": selected_names,
        "nh4_rmse": errors[0],
        "sno_rmse": errors[1],
    })


# ============================================================
# REPORT RESULTS
# ============================================================

nh4_values = np.array([
    r["nh4_rmse"]
    for r in results
])

sno_values = np.array([
    r["sno_rmse"]
    for r in results
])


best_nh4_index = np.argmin(nh4_values)
best_sno_index = np.argmin(sno_values)

best_nh4 = results[best_nh4_index]
best_sno = results[best_sno_index]


print()
print("=" * 70)
print("ESTIMATOR INTERFACE TEST")
print("=" * 70)


print()
print("Best NH4-N configuration:")
print("  ", ", ".join(best_nh4["names"]))
print("  NH4 RMSE:", f"{best_nh4['nh4_rmse']:.6f}")
print("  SNO RMSE:", f"{best_nh4['sno_rmse']:.6f}")


print()
print("Best SNO configuration:")
print("  ", ", ".join(best_sno["names"]))
print("  NH4 RMSE:", f"{best_sno['nh4_rmse']:.6f}")
print("  SNO RMSE:", f"{best_sno['sno_rmse']:.6f}")


print()
print("Across all 495 configurations:")
print(
    "  NH4 RMSE - mean:",
    f"{nh4_values.mean():.6f}"
)
print(
    "  NH4 RMSE - best:",
    f"{nh4_values.min():.6f}"
)
print(
    "  NH4 RMSE - worst:",
    f"{nh4_values.max():.6f}"
)

print(
    "  SNO RMSE - mean:",
    f"{sno_values.mean():.6f}"
)
print(
    "  SNO RMSE - best:",
    f"{sno_values.min():.6f}"
)
print(
    "  SNO RMSE - worst:",
    f"{sno_values.max():.6f}"
)


print()
print("=" * 70)
print("ESTIMATOR INTERFACE TEST COMPLETE")
print("=" * 70)