import os
import sys
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from dqn_data_loader import (
    eval_indices,
    train_indices,
    sensors,
    targets,
)

from ca_rdqn_training_environment import (
    CausalRDQNEnvironment,
    ACTION_TABLE,
    ACTION_DIM,
    OBSERVATION_DIM as ENV_OBSERVATION_DIM,
    STATE_HISTORY,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 22

TOTAL_SENSORS = 12

ACTIVE_SENSORS = 4

EVAL_START = int(
    eval_indices[0]
)

EVAL_END = int(
    eval_indices[-1]
)

HISTORY = 4

HIDDEN_DIM = 128

FC_DIM = 64


# ============================================================
# SENSOR NAMES
# ============================================================

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


assert len(
    SENSOR_NAMES
) == TOTAL_SENSORS


# ============================================================
# PATHS
# ============================================================

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_ca_rdqn",
)

CHECKPOINT_PATH = os.path.join(
    RESULT_DIR,
    "ca_rdqn_seed_22.pt",
)

RESULTS_PATH = os.path.join(
    RESULT_DIR,
    "ca_rdqn_seed_22_results.json",
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "ca_rdqn_seed_22_history.npz",
)


assert os.path.exists(
    CHECKPOINT_PATH
), CHECKPOINT_PATH

assert os.path.exists(
    RESULTS_PATH
), RESULTS_PATH

assert os.path.exists(
    HISTORY_PATH
), HISTORY_PATH


# ============================================================
# NETWORK
# ============================================================

class RecurrentDQN(nn.Module):

    def __init__(
        self,
        observation_dim=ENV_OBSERVATION_DIM,
        hidden_dim=HIDDEN_DIM,
        fc_dim=FC_DIM,
        action_dim=ACTION_DIM,
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=observation_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.q_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                fc_dim
            ),

            nn.ReLU(),

            nn.Linear(
                fc_dim,
                action_dim
            ),
        )


    def forward(
        self,
        x
    ):

        output, _ = self.gru(x)

        final_hidden = (
            output[:, -1, :]
        )

        return self.q_head(
            final_hidden
        )


# ============================================================
# LOAD SAVED HISTORY
# ============================================================

history = np.load(
    HISTORY_PATH
)


saved_eval_indices = history[
    "evaluation_indices"
]

saved_eval_actions = history[
    "evaluation_actions"
]

saved_eval_rewards = history[
    "evaluation_rewards"
]

saved_errors_nh4 = history[
    "evaluation_errors_nh4"
]

saved_errors_sno = history[
    "evaluation_errors_sno"
]

saved_predictions_nh4 = history[
    "evaluation_predictions_nh4"
]

saved_predictions_sno = history[
    "evaluation_predictions_sno"
]

saved_actual_nh4 = history[
    "evaluation_actual_nh4"
]

saved_actual_sno = history[
    "evaluation_actual_sno"
]


# ============================================================
# BASIC HISTORY CHECKS
# ============================================================

assert len(
    saved_eval_indices
) == 478

assert len(
    saved_eval_actions
) == 478

assert len(
    saved_eval_rewards
) == 478


assert np.array_equal(
    saved_eval_indices,
    eval_indices
)


assert np.isfinite(
    saved_eval_rewards
).all()

assert np.isfinite(
    saved_errors_nh4
).all()

assert np.isfinite(
    saved_errors_sno
).all()


print("=" * 70)
print(
    "STEP 71 — SEED 22 POLICY AUDIT"
)
print("=" * 70)

print(
    f"[INFO] Evaluation decisions: "
    f"{len(saved_eval_indices)}"
)

print(
    f"[INFO] Evaluation range: "
    f"{EVAL_START}–{EVAL_END}"
)

print(
    f"[INFO] Feasible actions: "
    f"{ACTION_DIM}"
)


# ============================================================
# ACTION CONSTRAINT AUDIT
# ============================================================

for action in saved_eval_actions:

    selected = ACTION_TABLE[
        int(action)
    ]

    assert len(
        selected
    ) == ACTIVE_SENSORS

    assert len(
        set(
            selected.tolist()
        )
    ) == ACTIVE_SENSORS


print(
    "[PASS] Every evaluation action activates "
    "exactly four distinct sensors."
)


# ============================================================
# ACTION FREQUENCY
# ============================================================

action_counter = Counter(
    int(action)
    for action
    in saved_eval_actions
)


print()
print(
    "TOP 20 SELECTED SENSOR COMBINATIONS"
)

print(
    "-" * 70
)


for rank, (
    action,
    count
) in enumerate(
    action_counter.most_common(20),
    start=1
):

    selected = ACTION_TABLE[
        action
    ]

    names = [
        SENSOR_NAMES[
            int(sensor_id)
        ]
        for sensor_id
        in selected
    ]

    percentage = (
        count
        /
        len(saved_eval_actions)
        *
        100.0
    )

    print(
        f"{rank:2d}. "
        f"Action {action:3d} | "
        f"{', '.join(names):45s} | "
        f"{count:3d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# SENSOR ACTIVATION FREQUENCY
# ============================================================

sensor_counts = np.zeros(
    TOTAL_SENSORS,
    dtype=np.int64
)


for action in saved_eval_actions:

    selected = ACTION_TABLE[
        int(action)
    ]

    for sensor_id in selected:

        sensor_counts[
            int(sensor_id)
        ] += 1


print()
print(
    "INDIVIDUAL SENSOR ACTIVATION FREQUENCY"
)

print(
    "-" * 70
)


sensor_order = np.argsort(
    -sensor_counts
)


for sensor_id in sensor_order:

    count = sensor_counts[
        sensor_id
    ]

    percentage = (
        count
        /
        len(saved_eval_actions)
        *
        100.0
    )

    print(
        f"{SENSOR_NAMES[sensor_id]:10s} : "
        f"{count:3d} / "
        f"{len(saved_eval_actions)} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# POLICY CONCENTRATION
# ============================================================

unique_actions = len(
    action_counter
)

top_1_share = (
    action_counter.most_common(1)[0][1]
    /
    len(saved_eval_actions)
    *
    100.0
)

top_5_count = sum(
    count
    for _, count
    in action_counter.most_common(5)
)

top_10_count = sum(
    count
    for _, count
    in action_counter.most_common(10)
)

top_5_share = (
    top_5_count
    /
    len(saved_eval_actions)
    *
    100.0
)

top_10_share = (
    top_10_count
    /
    len(saved_eval_actions)
    *
    100.0
)


print()
print(
    "POLICY CONCENTRATION"
)

print(
    "-" * 70
)

print(
    f"Unique actions: "
    f"{unique_actions}"
)

print(
    f"Top-1 action share: "
    f"{top_1_share:.2f}%"
)

print(
    f"Top-5 action share: "
    f"{top_5_share:.2f}%"
)

print(
    f"Top-10 action share: "
    f"{top_10_share:.2f}%"
)


# ============================================================
# REWARD DISTRIBUTION
# ============================================================

print()
print(
    "EVALUATION REWARD DISTRIBUTION"
)

print(
    "-" * 70
)

print(
    f"Mean:   "
    f"{np.mean(saved_eval_rewards):.6f}"
)

print(
    f"Std:    "
    f"{np.std(saved_eval_rewards):.6f}"
)

print(
    f"Median: "
    f"{np.median(saved_eval_rewards):.6f}"
)

print(
    f"Min:    "
    f"{np.min(saved_eval_rewards):.6f}"
)

print(
    f"Max:    "
    f"{np.max(saved_eval_rewards):.6f}"
)


# ============================================================
# TARGET ERROR DISTRIBUTIONS
# ============================================================

abs_nh4 = np.abs(
    saved_errors_nh4
)

abs_sno = np.abs(
    saved_errors_sno
)


print()
print(
    "TARGET ERROR DISTRIBUTIONS"
)

print(
    "-" * 70
)

print(
    "ΔNH4:"
)

print(
    f"  RMSE:   "
    f"{np.sqrt(np.mean(saved_errors_nh4 ** 2)):.6f}"
)

print(
    f"  MAE:    "
    f"{np.mean(abs_nh4):.6f}"
)

print(
    f"  Median: "
    f"{np.median(abs_nh4):.6f}"
)

print(
    f"  P90:    "
    f"{np.percentile(abs_nh4, 90):.6f}"
)

print(
    f"  P95:    "
    f"{np.percentile(abs_nh4, 95):.6f}"
)

print()

print(
    "ΔSNO:"
)

print(
    f"  RMSE:   "
    f"{np.sqrt(np.mean(saved_errors_sno ** 2)):.6f}"
)

print(
    f"  MAE:    "
    f"{np.mean(abs_sno):.6f}"
)

print(
    f"  Median: "
    f"{np.median(abs_sno):.6f}"
)

print(
    f"  P90:    "
    f"{np.percentile(abs_sno, 90):.6f}"
)

print(
    f"  P95:    "
    f"{np.percentile(abs_sno, 95):.6f}"
)


# ============================================================
# TARGET-SPECIFIC ACTION PERFORMANCE
# ============================================================

action_target_stats = {}

for action in sorted(
    np.unique(
        saved_eval_actions
    )
):

    mask = (
        saved_eval_actions
        ==
        action
    )

    action_target_stats[
        str(int(action))
    ] = {

        "count":
            int(np.sum(mask)),

        "nh4_rmse":
            float(
                np.sqrt(
                    np.mean(
                        saved_errors_nh4[mask]
                        ** 2
                    )
                )
            ),

        "sno_rmse":
            float(
                np.sqrt(
                    np.mean(
                        saved_errors_sno[mask]
                        ** 2
                    )
                )
            ),
    }


# ============================================================
# CHECKPOINT REPRODUCTION
# ============================================================

print()
print(
    "CHECKPOINT REPRODUCTION TEST"
)

print(
    "-" * 70
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)


assert (
    checkpoint["seed"]
    ==
    SEED
)


network = RecurrentDQN()

network.load_state_dict(
    checkpoint[
        "policy_state_dict"
    ]
)

network.eval()


# ------------------------------------------------------------
# Re-run greedy evaluation.
# ------------------------------------------------------------

nh4_weights_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "causal_dynamic_estimator",
    "causal_estimator_nh4_weights.npz",
)

sno_weights_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "causal_dynamic_estimator",
    "causal_estimator_sno_weights.npz",
)


assert os.path.exists(
    nh4_weights_path
)

assert os.path.exists(
    sno_weights_path
)


nh4_weights = np.load(
    nh4_weights_path
)

sno_weights = np.load(
    sno_weights_path
)


nh4_weights_tuple = (
    nh4_weights["W1"],
    nh4_weights["b1"],
    nh4_weights["W2"],
    nh4_weights["b2"],
    nh4_weights["W3"],
    nh4_weights["b3"],
)

sno_weights_tuple = (
    sno_weights["W1"],
    sno_weights["b1"],
    sno_weights["W2"],
    sno_weights["b2"],
    sno_weights["W3"],
    sno_weights["b3"],
)


eval_env = CausalRDQNEnvironment(
    start_index=EVAL_START,
    end_index=EVAL_END,
)


state = eval_env.reset()

done = False

reproduced_actions = []

reproduced_rewards = []

reproduced_nh4_errors = []

reproduced_sno_errors = []


while not done:

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)


    with torch.no_grad():

        q_values = network(
            state_tensor
        )

        action = int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )


    (
        next_state,
        reward,
        done,
        info,
    ) = eval_env.step(
        action,
        nh4_weights_tuple,
        sno_weights_tuple,
    )


    reproduced_actions.append(
        action
    )

    reproduced_rewards.append(
        reward
    )

    reproduced_nh4_errors.append(
        info["error_nh4"]
    )

    reproduced_sno_errors.append(
        info["error_sno"]
    )


    state = next_state


reproduced_actions = np.asarray(
    reproduced_actions,
    dtype=np.int64
)

reproduced_rewards = np.asarray(
    reproduced_rewards,
    dtype=np.float64
)

reproduced_nh4_errors = np.asarray(
    reproduced_nh4_errors,
    dtype=np.float64
)

reproduced_sno_errors = np.asarray(
    reproduced_sno_errors,
    dtype=np.float64
)


assert np.array_equal(
    reproduced_actions,
    saved_eval_actions
)


assert np.allclose(
    reproduced_rewards,
    saved_eval_rewards,
    atol=1e-8
)

assert np.allclose(
    reproduced_nh4_errors,
    saved_errors_nh4,
    atol=1e-8
)

assert np.allclose(
    reproduced_sno_errors,
    saved_errors_sno,
    atol=1e-8
)


print(
    "[PASS] Saved checkpoint reproduces "
    "the exact evaluation action sequence."
)

print(
    "[PASS] Saved rewards reproduced."
)

print(
    "[PASS] Saved ΔNH4 errors reproduced."
)

print(
    "[PASS] Saved ΔSNO errors reproduced."
)


# ============================================================
# SAVE AUDIT
# ============================================================

audit_results = {

    "seed": SEED,

    "evaluation_decisions":
        int(len(saved_eval_actions)),

    "unique_actions":
        int(unique_actions),

    "top_1_share_percent":
        float(top_1_share),

    "top_5_share_percent":
        float(top_5_share),

    "top_10_share_percent":
        float(top_10_share),

    "sensor_activation_counts": {
        SENSOR_NAMES[i]:
            int(sensor_counts[i])
        for i in range(
            TOTAL_SENSORS
        )
    },

    "nh4_rmse":
        float(
            np.sqrt(
                np.mean(
                    saved_errors_nh4 ** 2
                )
            )
        ),

    "sno_rmse":
        float(
            np.sqrt(
                np.mean(
                    saved_errors_sno ** 2
                )
            )
        ),

    "checkpoint_reproduction":
        True,

    "exact_four_constraint":
        True,

    "evaluation_only":
        True,

    "action_target_statistics":
        action_target_stats,
}


audit_path = os.path.join(
    RESULT_DIR,
    "ca_rdqn_seed_22_policy_audit.json"
)


with open(
    audit_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit_results,
        f,
        indent=2
    )


assert os.path.exists(
    audit_path
)


# ============================================================
# FINAL
# ============================================================

print()
print(
    "[PASS] Seed 22 policy audit completed."
)

print(
    f"[INFO] Audit saved to:"
)

print(
    f"       {audit_path}"
)

print()
print("=" * 70)
print(
    "STEP 71 COMPLETED"
)
print("=" * 70)