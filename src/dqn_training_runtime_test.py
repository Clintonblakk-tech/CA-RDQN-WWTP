import numpy as np

print("=" * 70)
print("STEP 26 — DQN TRAINING RUNTIME TEST")
print("=" * 70)

# ------------------------------------------------------------
# 1. Import PyTorch
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print("[PASS] PyTorch imported successfully.")
    print(f"[INFO] PyTorch version: {torch.__version__}")

except Exception as e:
    print("[FAIL] PyTorch import failed.")
    print(f"[ERROR] {type(e).__name__}: {e}")
    raise SystemExit(1)


# ------------------------------------------------------------
# 2. Reproducibility
# ------------------------------------------------------------
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print("[PASS] Random seeds initialized.")


# ------------------------------------------------------------
# 3. DQN configuration
# ------------------------------------------------------------
STATE_DIM = 96
ACTION_DIM = 495

HIDDEN_1 = 128
HIDDEN_2 = 64

GAMMA = 0.99
LEARNING_RATE = 1e-3


# ------------------------------------------------------------
# 4. Define DQN network
# ------------------------------------------------------------
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, action_dim)
        )

    def forward(self, x):
        return self.network(x)


policy_net = DQN(STATE_DIM, ACTION_DIM)
target_net = DQN(STATE_DIM, ACTION_DIM)

print("[PASS] Policy network created.")
print("[PASS] Target network created.")


# ------------------------------------------------------------
# 5. Synchronize target network
# ------------------------------------------------------------
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

print("[PASS] Target network synchronized.")


# ------------------------------------------------------------
# 6. Optimizer
# ------------------------------------------------------------
optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE
)

print("[PASS] Adam optimizer created.")


# ------------------------------------------------------------
# 7. Synthetic replay batch
# ------------------------------------------------------------
BATCH_SIZE = 64

states = torch.tensor(
    np.random.randn(BATCH_SIZE, STATE_DIM),
    dtype=torch.float32
)

next_states = torch.tensor(
    np.random.randn(BATCH_SIZE, STATE_DIM),
    dtype=torch.float32
)

actions = torch.tensor(
    np.random.randint(0, ACTION_DIM, size=BATCH_SIZE),
    dtype=torch.long
)

rewards = torch.tensor(
    np.random.randn(BATCH_SIZE),
    dtype=torch.float32
)

dones = torch.zeros(
    BATCH_SIZE,
    dtype=torch.float32
)

print("[PASS] Synthetic replay batch created.")


# ------------------------------------------------------------
# 8. Forward pass
# ------------------------------------------------------------
q_values = policy_net(states)

print("[PASS] Policy-network forward pass completed.")
print(f"[INFO] Q-value tensor shape: {tuple(q_values.shape)}")


# ------------------------------------------------------------
# 9. Select Q-values for sampled actions
# ------------------------------------------------------------
current_q = q_values.gather(
    1,
    actions.unsqueeze(1)
).squeeze(1)

print("[PASS] Action-specific Q-values extracted.")


# ------------------------------------------------------------
# 10. Bellman target
# ------------------------------------------------------------
with torch.no_grad():

    next_q = target_net(next_states).max(
        dim=1
    ).values

    target_q = rewards + GAMMA * (1.0 - dones) * next_q

print("[PASS] Bellman targets calculated.")


# ------------------------------------------------------------
# 11. Loss
# ------------------------------------------------------------
loss_fn = nn.SmoothL1Loss()

loss = loss_fn(
    current_q,
    target_q
)

print("[PASS] DQN loss calculated.")
print(f"[INFO] Initial loss: {loss.item():.6f}")


# ------------------------------------------------------------
# 12. Backpropagation
# ------------------------------------------------------------
optimizer.zero_grad()

loss.backward()

optimizer.step()

print("[PASS] Backpropagation completed.")
print("[PASS] Optimizer update completed.")


# ------------------------------------------------------------
# 13. Target-network update
# ------------------------------------------------------------
target_net.load_state_dict(
    policy_net.state_dict()
)

print("[PASS] Target-network update completed.")


# ------------------------------------------------------------
# 14. Final validation
# ------------------------------------------------------------
with torch.no_grad():

    test_state = torch.tensor(
        np.random.randn(1, STATE_DIM),
        dtype=torch.float32
    )

    test_q = policy_net(test_state)

assert test_q.shape == (1, ACTION_DIM)

assert torch.isfinite(test_q).all()

print("[PASS] Final network-output validation passed.")

print("=" * 70)
print("STEP 26 COMPLETED")
print("=" * 70)