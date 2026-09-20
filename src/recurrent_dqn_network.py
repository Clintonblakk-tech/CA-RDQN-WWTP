import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

OBSERVATION_DIM = 24
HISTORY = 4

HIDDEN_DIM = 128
FC_DIM = 64

ACTION_DIM = 495


# ============================================================
# RECURRENT DQN
# ============================================================

class RecurrentDQN(nn.Module):

    def __init__(
        self,
        observation_dim=OBSERVATION_DIM,
        hidden_dim=HIDDEN_DIM,
        fc_dim=FC_DIM,
        action_dim=ACTION_DIM,
    ):

        super().__init__()

        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.action_dim = action_dim

        # ----------------------------------------------------
        # Temporal encoder
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=observation_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Q-value head
        # ----------------------------------------------------

        self.q_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                fc_dim
            ),

            nn.ReLU(),

            nn.Linear(
                fc_dim,
                action_dim
            )
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # Expected:
        #
        # x.shape =
        # [batch, HISTORY, OBSERVATION_DIM]
        #

        assert x.ndim == 3

        assert x.shape[1] == HISTORY

        assert x.shape[2] == (
            self.observation_dim
        )

        assert torch.isfinite(
            x
        ).all()

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        sequence_output, hidden = (
            self.gru(x)
        )

        # Use the final temporal representation.

        final_hidden = (
            sequence_output[:, -1, :]
        )

        # ----------------------------------------------------
        # Q-values
        # ----------------------------------------------------

        q_values = self.q_head(
            final_hidden
        )

        assert q_values.shape[1] == (
            self.action_dim
        )

        assert torch.isfinite(
            q_values
        ).all()

        return q_values


# ============================================================
# VALIDATION
# ============================================================

print("=" * 70)
print("STEP 61 — RECURRENT DQN NETWORK VALIDATION")
print("=" * 70)


print(
    f"[INFO] Observation dimension: "
    f"{OBSERVATION_DIM}"
)

print(
    f"[INFO] History length: "
    f"{HISTORY}"
)

print(
    f"[INFO] GRU hidden dimension: "
    f"{HIDDEN_DIM}"
)

print(
    f"[INFO] Fully connected dimension: "
    f"{FC_DIM}"
)

print(
    f"[INFO] Action dimension: "
    f"{ACTION_DIM}"
)


# ============================================================
# MODEL
# ============================================================

model = RecurrentDQN()

print(
    "[PASS] Recurrent DQN initialized."
)


# ============================================================
# PARAMETER CHECK
# ============================================================

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

assert parameter_count > 0

print(
    f"[INFO] Trainable parameters: "
    f"{parameter_count}"
)


# ============================================================
# SINGLE SAMPLE FORWARD PASS
# ============================================================

torch.manual_seed(42)

single_input = torch.randn(
    1,
    HISTORY,
    OBSERVATION_DIM,
    dtype=torch.float32
)

single_output = model(
    single_input
)

assert single_output.shape == (
    1,
    ACTION_DIM
)

assert torch.isfinite(
    single_output
).all()

print(
    "[PASS] Single-sample forward pass verified."
)

print(
    f"[INFO] Q-value shape: "
    f"{tuple(single_output.shape)}"
)


# ============================================================
# BATCH FORWARD PASS
# ============================================================

batch_input = torch.randn(
    32,
    HISTORY,
    OBSERVATION_DIM,
    dtype=torch.float32
)

batch_output = model(
    batch_input
)

assert batch_output.shape == (
    32,
    ACTION_DIM
)

assert torch.isfinite(
    batch_output
).all()

print(
    "[PASS] Batch forward pass verified."
)

print(
    f"[INFO] Batch Q-value shape: "
    f"{tuple(batch_output.shape)}"
)


# ============================================================
# GREEDY ACTION TEST
# ============================================================

greedy_actions = torch.argmax(
    batch_output,
    dim=1
)

assert greedy_actions.shape == (
    32,
)

assert torch.all(
    greedy_actions >= 0
)

assert torch.all(
    greedy_actions < ACTION_DIM
)

print(
    "[PASS] Greedy action selection verified."
)


# ============================================================
# BACKPROPAGATION TEST
# ============================================================

target = torch.zeros_like(
    batch_output
)

loss_function = nn.MSELoss()

loss = loss_function(
    batch_output,
    target
)

assert torch.isfinite(
    loss
)

loss.backward()

gradient_count = 0

for parameter in model.parameters():

    if parameter.grad is not None:

        assert torch.isfinite(
            parameter.grad
        ).all()

        gradient_count += 1

assert gradient_count > 0

print(
    "[PASS] Backpropagation verified."
)

print(
    f"[INFO] Parameters with gradients: "
    f"{gradient_count}"
)


# ============================================================
# OPTIMIZER TEST
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

optimizer.step()

optimizer.zero_grad()

print(
    "[PASS] Adam optimizer update verified."
)


# ============================================================
# REPRODUCIBILITY TEST
# ============================================================

torch.manual_seed(123)

model_a = RecurrentDQN()

test_input = torch.randn(
    4,
    HISTORY,
    OBSERVATION_DIM
)

output_a = model_a(
    test_input
)

torch.manual_seed(123)

model_b = RecurrentDQN()

output_b = model_b(
    test_input
)

assert torch.allclose(
    output_a,
    output_b
)

print(
    "[PASS] Recurrent DQN reproducibility verified."
)


# ============================================================
# FINAL
# ============================================================

print()
print(
    "[PASS] 24-D sequential observations verified."
)

print(
    "[PASS] GRU temporal representation verified."
)

print(
    "[PASS] 495-action Q-value output verified."
)

print(
    "[PASS] Gradient flow verified."
)

print(
    "[PASS] Optimizer update verified."
)

print("=" * 70)
print("STEP 61 COMPLETED")
print("=" * 70)