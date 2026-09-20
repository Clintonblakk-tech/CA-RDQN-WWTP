import os
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ESTIMATOR_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "frozen_estimator"
)

NH4_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_nh4.joblib"
)

SNO_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_sno.joblib"
)


print("=" * 70)
print("STEP 57 — FROZEN ESTIMATOR ARCHITECTURE INSPECTION")
print("=" * 70)


nh4_model = joblib.load(NH4_PATH)
sno_model = joblib.load(SNO_PATH)


for name, model in [
    ("NH4", nh4_model),
    ("SNO", sno_model),
]:

    print()
    print(f"[INFO] {name} estimator")
    print("-" * 70)

    print(
        f"[INFO] Input dimension: "
        f"{model.n_features_in_}"
    )

    print(
        f"[INFO] Output dimension: "
        f"{model.n_outputs_}"
    )

    print(
        f"[INFO] Hidden-layer sizes: "
        f"{model.hidden_layer_sizes}"
    )

    print(
        f"[INFO] Number of layers: "
        f"{model.n_layers_}"
    )

    print(
        f"[INFO] Activation: "
        f"{model.activation}"
    )

    print(
        f"[INFO] Output activation: "
        f"{model.out_activation_}"
    )

    print(
        f"[INFO] Number of weight matrices: "
        f"{len(model.coefs_)}"
    )

    for i, (weights, biases) in enumerate(
        zip(model.coefs_, model.intercepts_)
    ):

        print(
            f"[INFO] Layer {i + 1} weights: "
            f"{weights.shape}"
        )

        print(
            f"[INFO] Layer {i + 1} biases: "
            f"{biases.shape}"
        )

        assert np.isfinite(weights).all()
        assert np.isfinite(biases).all()

    print(
        f"[PASS] {name} estimator parameters "
        f"are finite."
    )


assert nh4_model.n_features_in_ == 120
assert sno_model.n_features_in_ == 120

assert nh4_model.hidden_layer_sizes == (128, 64)
assert sno_model.hidden_layer_sizes == (128, 64)

assert nh4_model.activation == "relu"
assert sno_model.activation == "relu"

print()
print("[PASS] NH4 architecture verified.")
print("[PASS] SNO architecture verified.")
print("[PASS] 120-D input verified.")
print("[PASS] 128 -> 64 hidden layers verified.")
print("[PASS] ReLU activation verified.")

print("=" * 70)
print("STEP 57 COMPLETED")
print("=" * 70)