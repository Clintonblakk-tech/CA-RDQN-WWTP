import os
import sys
import numpy as np
import torch
import torch.nn as nn


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

NH4_WEIGHTS_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_nh4_weights.npz"
)

SNO_WEIGHTS_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_sno_weights.npz"
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 120
HIDDEN_1 = 128
HIDDEN_2 = 64
OUTPUT_DIM = 1


# ============================================================
# MODEL
# ============================================================

class FrozenEstimator(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD NUMERICAL PARAMETERS
# ============================================================

print("=" * 70)
print("STEP 58B — PYTORCH FROZEN ESTIMATOR VALIDATION")
print("=" * 70)

nh4_weights = np.load(
    NH4_WEIGHTS_PATH
)

sno_weights = np.load(
    SNO_WEIGHTS_PATH
)

print("[PASS] NH4 numerical parameters loaded.")
print("[PASS] SNO numerical parameters loaded.")


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(weights):

    model = FrozenEstimator()

    with torch.no_grad():

        model.network[0].weight.copy_(
            torch.tensor(
                weights["W1"].T,
                dtype=torch.float32
            )
        )

        model.network[0].bias.copy_(
            torch.tensor(
                weights["b1"],
                dtype=torch.float32
            )
        )

        model.network[2].weight.copy_(
            torch.tensor(
                weights["W2"].T,
                dtype=torch.float32
            )
        )

        model.network[2].bias.copy_(
            torch.tensor(
                weights["b2"],
                dtype=torch.float32
            )
        )

        model.network[4].weight.copy_(
            torch.tensor(
                weights["W3"].T,
                dtype=torch.float32
            )
        )

        model.network[4].bias.copy_(
            torch.tensor(
                weights["b3"],
                dtype=torch.float32
            )
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


nh4_model = build_model(
    nh4_weights
)

sno_model = build_model(
    sno_weights
)


# ============================================================
# ARCHITECTURE CHECK
# ============================================================

assert nh4_model.network[0].in_features == 120
assert nh4_model.network[0].out_features == 128
assert nh4_model.network[2].in_features == 128
assert nh4_model.network[2].out_features == 64
assert nh4_model.network[4].in_features == 64
assert nh4_model.network[4].out_features == 1

assert sno_model.network[0].in_features == 120
assert sno_model.network[0].out_features == 128
assert sno_model.network[2].in_features == 128
assert sno_model.network[2].out_features == 64
assert sno_model.network[4].in_features == 64
assert sno_model.network[4].out_features == 1

print("[PASS] NH4 PyTorch architecture verified.")
print("[PASS] SNO PyTorch architecture verified.")


# ============================================================
# FROZEN PARAMETER CHECK
# ============================================================

for model in [
    nh4_model,
    sno_model
]:

    for parameter in model.parameters():

        assert parameter.requires_grad is False

        assert torch.isfinite(
            parameter
        ).all()


print("[PASS] NH4 parameters frozen.")
print("[PASS] SNO parameters frozen.")
print("[PASS] All PyTorch parameters are finite.")


# ============================================================
# NUMERICAL FORWARD-PASS TEST
# ============================================================

rng = np.random.default_rng(42)

test_input = rng.normal(
    size=(100, INPUT_DIM)
).astype(np.float32)

test_tensor = torch.tensor(
    test_input,
    dtype=torch.float32
)

with torch.no_grad():

    nh4_prediction = (
        nh4_model(test_tensor)
        .cpu()
        .numpy()
        .reshape(-1)
    )

    sno_prediction = (
        sno_model(test_tensor)
        .cpu()
        .numpy()
        .reshape(-1)
    )


assert nh4_prediction.shape == (100,)
assert sno_prediction.shape == (100,)

assert np.isfinite(
    nh4_prediction
).all()

assert np.isfinite(
    sno_prediction
).all()


print("[PASS] NH4 PyTorch forward pass completed.")
print("[PASS] SNO PyTorch forward pass completed.")
print("[PASS] Prediction dimensions verified.")
print("[PASS] Predictions are finite.")


# ============================================================
# SAVE PYTORCH STATE DICTIONARIES
# ============================================================

nh4_pt_path = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_nh4_torch.pt"
)

sno_pt_path = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_sno_torch.pt"
)

torch.save(
    nh4_model.state_dict(),
    nh4_pt_path
)

torch.save(
    sno_model.state_dict(),
    sno_pt_path
)


# ============================================================
# RELOAD TEST
# ============================================================

nh4_reload = FrozenEstimator()
sno_reload = FrozenEstimator()

nh4_reload.load_state_dict(
    torch.load(
        nh4_pt_path,
        map_location="cpu"
    )
)

sno_reload.load_state_dict(
    torch.load(
        sno_pt_path,
        map_location="cpu"
    )
)

nh4_reload.eval()
sno_reload.eval()


with torch.no_grad():

    nh4_reload_prediction = (
        nh4_reload(test_tensor)
        .cpu()
        .numpy()
        .reshape(-1)
    )

    sno_reload_prediction = (
        sno_reload(test_tensor)
        .cpu()
        .numpy()
        .reshape(-1)
    )


assert np.allclose(
    nh4_prediction,
    nh4_reload_prediction,
    rtol=1e-6,
    atol=1e-7
)

assert np.allclose(
    sno_prediction,
    sno_reload_prediction,
    rtol=1e-6,
    atol=1e-7
)


print("[PASS] NH4 PyTorch reload verified.")
print("[PASS] SNO PyTorch reload verified.")
print("[PASS] PyTorch prediction reproducibility verified.")


# ============================================================
# FINAL
# ============================================================

print()
print(f"[INFO] NH4 PyTorch model:")
print(f"       {nh4_pt_path}")

print(f"[INFO] SNO PyTorch model:")
print(f"       {sno_pt_path}")

print("=" * 70)
print("STEP 58B COMPLETED")
print("=" * 70)