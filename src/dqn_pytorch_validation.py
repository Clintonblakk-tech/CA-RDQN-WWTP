import torch
import torch.nn as nn


# ============================================================
# STEP 22 — PYTORCH / DQN COMPATIBILITY VALIDATION
# ============================================================

STATE_DIM = 96
ACTION_DIM = 495

print("\n" + "=" * 70)
print("STEP 22 — PYTORCH / DQN COMPATIBILITY VALIDATION")
print("=" * 70)


# ============================================================
# PyTorch version
# ============================================================

print("\nPyTorch:")
print(f"  Version: {torch.__version__}")


# ============================================================
# Device
# ============================================================

device = torch.device("cpu")

print("\nDevice:")
print(f"  Selected device: {device}")


# ============================================================
# Define DQN
# ============================================================

class DQN(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim),
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Construct network
# ============================================================

print("\nConstructing DQN...")

policy_net = DQN(
    STATE_DIM,
    ACTION_DIM,
).to(device)

target_net = DQN(
    STATE_DIM,
    ACTION_DIM,
).to(device)

print("  Policy network constructed.")
print("  Target network constructed.")


# ============================================================
# Synthetic state
# ============================================================

print("\nTesting forward pass...")

test_state = torch.zeros(
    (1, STATE_DIM),
    dtype=torch.float32,
    device=device,
)

with torch.no_grad():

    q_values = policy_net(
        test_state
    )

print(
    f"  Input shape : "
    f"{tuple(test_state.shape)}"
)

print(
    f"  Output shape: "
    f"{tuple(q_values.shape)}"
)

assert q_values.shape == (
    1,
    ACTION_DIM,
)


# ============================================================
# Greedy action
# ============================================================

print("\nTesting action selection...")

selected_action = torch.argmax(
    q_values,
    dim=1,
).item()

print(
    f"  Selected action index: "
    f"{selected_action}"
)

assert 0 <= selected_action < ACTION_DIM


# ============================================================
# Training operation
# ============================================================

print("\nTesting backward pass...")

optimizer = torch.optim.Adam(
    policy_net.parameters(),
    lr=1e-3,
)

target = torch.zeros_like(
    q_values
)

loss_function = nn.MSELoss()

optimizer.zero_grad()

q_values = policy_net(
    test_state
)

loss = loss_function(
    q_values,
    target,
)

loss.backward()

optimizer.step()

print(
    f"  Loss: {loss.item():.8f}"
)

print("  Backward pass completed.")
print("  Optimizer step completed.")


# ============================================================
# Target-network synchronization
# ============================================================

print("\nTesting target-network synchronization...")

target_net.load_state_dict(
    policy_net.state_dict()
)

print(
    "  Target network synchronized "
    "successfully."
)


# ============================================================
# Final audit
# ============================================================

print("\n" + "=" * 70)
print("STEP 22 RESULTS")
print("=" * 70)

print("[PASS] PyTorch imported successfully.")
print("[PASS] DQN policy network constructed.")
print("[PASS] Target network constructed.")
print("[PASS] 96-dimensional state accepted.")
print("[PASS] 495 Q-values produced.")
print("[PASS] Valid action index selected.")
print("[PASS] Backward pass completed.")
print("[PASS] Optimizer step completed.")
print("[PASS] Target-network synchronization completed.")

print("\n" + "=" * 70)
print("STEP 22 COMPLETED")
print("=" * 70)