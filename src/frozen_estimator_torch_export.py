import os
import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

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

NH4_WEIGHTS_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_nh4_weights.npz"
)

SNO_WEIGHTS_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_sno_weights.npz"
)


# ============================================================
# EXPORT FUNCTION
# ============================================================

def export_weights(model, output_path):

    assert model.n_features_in_ == 120
    assert model.hidden_layer_sizes == (128, 64)
    assert model.activation == "relu"
    assert model.out_activation_ == "identity"

    assert len(model.coefs_) == 3
    assert len(model.intercepts_) == 3

    for weights in model.coefs_:
        assert np.isfinite(weights).all()

    for biases in model.intercepts_:
        assert np.isfinite(biases).all()

    np.savez(
        output_path,
        W1=model.coefs_[0].astype(np.float32),
        b1=model.intercepts_[0].astype(np.float32),
        W2=model.coefs_[1].astype(np.float32),
        b2=model.intercepts_[1].astype(np.float32),
        W3=model.coefs_[2].astype(np.float32),
        b3=model.intercepts_[2].astype(np.float32),
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("STEP 58A — FROZEN ESTIMATOR WEIGHT EXPORT")
print("=" * 70)


# ------------------------------------------------------------
# Load validated sklearn estimators
# ------------------------------------------------------------

nh4_model = joblib.load(NH4_PATH)
sno_model = joblib.load(SNO_PATH)

print("[PASS] NH4 estimator loaded.")
print("[PASS] SNO estimator loaded.")


# ------------------------------------------------------------
# Verify architectures
# ------------------------------------------------------------

for name, model in [
    ("NH4", nh4_model),
    ("SNO", sno_model),
]:

    assert model.n_features_in_ == 120
    assert model.hidden_layer_sizes == (128, 64)
    assert model.activation == "relu"
    assert model.out_activation_ == "identity"

    print(
        f"[PASS] {name} architecture verified."
    )


# ------------------------------------------------------------
# Export weights
# ------------------------------------------------------------

export_weights(
    nh4_model,
    NH4_WEIGHTS_PATH
)

export_weights(
    sno_model,
    SNO_WEIGHTS_PATH
)

print("[PASS] NH4 weights exported.")
print("[PASS] SNO weights exported.")


# ------------------------------------------------------------
# Reload exported numerical parameters
# ------------------------------------------------------------

nh4_weights = np.load(
    NH4_WEIGHTS_PATH
)

sno_weights = np.load(
    SNO_WEIGHTS_PATH
)


# ------------------------------------------------------------
# Verify shapes
# ------------------------------------------------------------

expected_shapes = {
    "W1": (120, 128),
    "b1": (128,),
    "W2": (128, 64),
    "b2": (64,),
    "W3": (64, 1),
    "b3": (1,),
}

for name, shape in expected_shapes.items():

    assert nh4_weights[name].shape == shape
    assert sno_weights[name].shape == shape

    assert np.isfinite(
        nh4_weights[name]
    ).all()

    assert np.isfinite(
        sno_weights[name]
    ).all()


print("[PASS] NH4 weight shapes verified.")
print("[PASS] SNO weight shapes verified.")
print("[PASS] All exported parameters are finite.")


# ------------------------------------------------------------
# Verify numerical parameter preservation
# ------------------------------------------------------------

for name in expected_shapes:

    original_nh4 = (
        nh4_model.coefs_[int(name[1]) - 1]
        if name.startswith("W")
        else nh4_model.intercepts_[int(name[1]) - 1]
    )

    original_sno = (
        sno_model.coefs_[int(name[1]) - 1]
        if name.startswith("W")
        else sno_model.intercepts_[int(name[1]) - 1]
    )

    assert np.allclose(
        original_nh4,
        nh4_weights[name],
        rtol=1e-6,
        atol=1e-7
    )

    assert np.allclose(
        original_sno,
        sno_weights[name],
        rtol=1e-6,
        atol=1e-7
    )


print("[PASS] NH4 parameter preservation verified.")
print("[PASS] SNO parameter preservation verified.")


print()
print(f"[INFO] NH4 weights:")
print(f"       {NH4_WEIGHTS_PATH}")

print(f"[INFO] SNO weights:")
print(f"       {SNO_WEIGHTS_PATH}")

print("=" * 70)
print("STEP 58A COMPLETED")
print("=" * 70)