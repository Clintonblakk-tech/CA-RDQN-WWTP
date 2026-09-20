import os
import sys
import random
import json

import numpy as np
import torch
import torch.nn as nn

from itertools import combinations


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
# DATA
# ============================================================

from dqn_data_loader import (
    sensors,
    targets,
    train_indices,
    eval_indices,
    HISTORY,
    SENSOR_COUNT,
)


# ============================================================
# ENVIRONMENT
# ============================================================

from ca_rdqn_training_environment import (
    CausalRDQNEnvironment,
    ACTION_TABLE,
    ACTION_DIM,
    OBSERVATION_DIM,
    STATE_HISTORY,
)


# ============================================================
# FINAL LOCKED CONFIGURATION
# ============================================================

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=33)
args = parser.parse_args()

SEED = args.seed

GAMMA = 0.99

LEARNING_RATE = 1e-3

BATCH_SIZE = 64

REPLAY_CAPACITY = 50_000

WARMUP_STEPS = 64

TARGET_UPDATE_FREQUENCY = 500

EPSILON_START = 1.0

EPSILON_END = 0.05

EPSILON_DECAY_STEPS = 20_000

TRAIN_EPISODES = 24

FORECAST_STEPS = 2

SENSOR_BUDGET = 4

TOTAL_SENSORS = 12

HIDDEN_DIM = 128

FC_DIM = 64

GRADIENT_CLIP = 10.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_START = int(
    train_indices[0]
)

TRAIN_END = int(
    train_indices[-1]
)

EVAL_START = int(
    eval_indices[0]
)

EVAL_END = int(
    eval_indices[-1]
)


assert TRAIN_END < EVAL_START


# ============================================================
# ACTION SPACE
# ============================================================

assert ACTION_TABLE.shape == (
    ACTION_DIM,
    SENSOR_BUDGET
)

assert ACTION_DIM == 495

assert TOTAL_SENSORS == 12

assert SENSOR_BUDGET == 4


# ============================================================
# NETWORK
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

        assert x.ndim == 3

        assert x.shape[1] == STATE_HISTORY

        assert x.shape[2] == OBSERVATION_DIM

        output, _ = self.gru(x)

        final_hidden = (
            output[:, -1, :]
        )

        q_values = self.q_head(
            final_hidden
        )

        assert q_values.shape[1] == ACTION_DIM

        return q_values


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity
    ):

        self.capacity = int(
            capacity
        )

        self.buffer = []

        self.position = 0


    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        transition = (
            np.asarray(
                state,
                dtype=np.float32
            ),

            int(action),

            float(reward),

            np.asarray(
                next_state,
                dtype=np.float32
            ),

            bool(done),
        )


        if len(
            self.buffer
        ) < self.capacity:

            self.buffer.append(
                transition
            )

        else:

            self.buffer[
                self.position
            ] = transition


        self.position = (
            self.position + 1
        ) % self.capacity


    def sample(
        self,
        batch_size
    ):

        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
        )

        batch = [
            self.buffer[i]
            for i in indices
        ]


        states = np.stack(
            [
                item[0]
                for item in batch
            ]
        )

        actions = np.asarray(
            [
                item[1]
                for item in batch
            ],
            dtype=np.int64
        )

        rewards = np.asarray(
            [
                item[2]
                for item in batch
            ],
            dtype=np.float32
        )

        next_states = np.stack(
            [
                item[3]
                for item in batch
            ]
        )

        dones = np.asarray(
            [
                item[4]
                for item in batch
            ],
            dtype=np.float32
        )


        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )


    def __len__(
        self
    ):

        return len(
            self.buffer
        )


# ============================================================
# EPSILON
# ============================================================

def epsilon_by_step(
    step
):

    fraction = min(
        step /
        EPSILON_DECAY_STEPS,
        1.0
    )

    return float(
        EPSILON_START
        +
        fraction
        *
        (
            EPSILON_END
            -
            EPSILON_START
        )
    )


# ============================================================
# ACTION SELECTION
# ============================================================

def select_action(
    network,
    state,
    epsilon,
):

    if random.random() < epsilon:

        return random.randrange(
            ACTION_DIM
        )


    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)


    with torch.no_grad():

        q_values = network(
            state_tensor
        )


    return int(
        torch.argmax(
            q_values,
            dim=1
        ).item()
    )


# ============================================================
# ESTIMATOR WEIGHTS
# ============================================================

def load_estimator_weights(
    filename
):

    path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "causal_dynamic_estimator",
        filename
    )

    assert os.path.exists(
        path
    ), path


    data = np.load(
        path
    )


    return (
        data["W1"],
        data["b1"],
        data["W2"],
        data["b2"],
        data["W3"],
        data["b3"],
    )


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_model(
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
):

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = replay_buffer.sample(
        BATCH_SIZE
    )


    states = torch.tensor(
        states,
        dtype=torch.float32
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    next_states = torch.tensor(
        next_states,
        dtype=torch.float32
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Online Q-values
    # --------------------------------------------------------

    q_values = policy_net(
        states
    )


    selected_q = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)


    # --------------------------------------------------------
    # Double-DQN target
    # --------------------------------------------------------

    with torch.no_grad():

        next_policy_q = policy_net(
            next_states
        )

        next_actions = torch.argmax(
            next_policy_q,
            dim=1,
            keepdim=True
        )


        next_target_q = target_net(
            next_states
        )


        next_selected_q = (
            next_target_q
            .gather(
                1,
                next_actions
            )
            .squeeze(1)
        )


        bellman_target = (
            rewards
            +
            GAMMA
            *
            (
                1.0 - dones
            )
            *
            next_selected_q
        )


    # --------------------------------------------------------
    # Huber loss
    # --------------------------------------------------------

    loss = nn.SmoothL1Loss()(
        selected_q,
        bellman_target
    )


    assert torch.isfinite(
        loss
    )


    optimizer.zero_grad()

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=GRADIENT_CLIP
    )


    optimizer.step()


    return float(
        loss.detach().cpu().item()
    )


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================

def create_environment(
    start_index,
    end_index
):

    return CausalRDQNEnvironment(
        start_index=int(
            start_index
        ),
        end_index=int(
            end_index
        ),
    )


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print(
    "STEP 68 — FINAL SINGLE-SEED CA-RDQN TRAINING"
)
print("=" * 70)

print(
    f"[INFO] Seed: {SEED}"
)

print(
    f"[INFO] Training episodes: "
    f"{TRAIN_EPISODES}"
)

print(
    f"[INFO] Training range: "
    f"{TRAIN_START}–{TRAIN_END}"
)

print(
    f"[INFO] Held-out evaluation range: "
    f"{EVAL_START}–{EVAL_END}"
)

print(
    f"[INFO] Sensors: "
    f"{TOTAL_SENSORS}"
)

print(
    f"[INFO] Active sensors: "
    f"{SENSOR_BUDGET}"
)

print(
    f"[INFO] Feasible actions: "
    f"{ACTION_DIM}"
)


# ============================================================
# LOAD ESTIMATORS
# ============================================================

nh4_weights = load_estimator_weights(
    "causal_estimator_nh4_weights.npz"
)

sno_weights = load_estimator_weights(
    "causal_estimator_sno_weights.npz"
)


print(
    "[PASS] Causal estimators loaded."
)


# ============================================================
# NETWORKS
# ============================================================

policy_net = RecurrentDQN()

target_net = RecurrentDQN()

target_net.load_state_dict(
    policy_net.state_dict()
)

target_net.eval()


optimizer = torch.optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE
)


print(
    "[PASS] Policy network initialized."
)

print(
    "[PASS] Target network initialized."
)

print(
    "[PASS] Optimizer initialized."
)


# ============================================================
# REPLAY
# ============================================================

replay_buffer = ReplayBuffer(
    REPLAY_CAPACITY
)


# ============================================================
# HISTORY ARRAYS
# ============================================================

episode_rewards = []

episode_losses = []

episode_action_counts = []

episode_lengths = []

global_step = 0

all_training_rewards = []

all_training_actions = []

all_training_indices = []


# ============================================================
# EPISODE LOOP
# ============================================================

for episode in range(
    TRAIN_EPISODES
):

    env = create_environment(
        TRAIN_START,
        TRAIN_END
    )


    state = env.reset()

    done = False

    episode_reward = 0.0

    episode_losses_current = []

    episode_actions = []

    episode_indices = []


    while not done:

        current_index = (
            env.current_index
        )


        # ----------------------------------------------------
        # Safety: training may never enter held-out data.
        # ----------------------------------------------------

        assert (
            current_index
            <=
            TRAIN_END
        )


        epsilon = epsilon_by_step(
            global_step
        )


        action = select_action(
            policy_net,
            state,
            epsilon
        )


        assert (
            0 <= action < ACTION_DIM
        )


        selected = ACTION_TABLE[
            action
        ]


        assert len(
            selected
        ) == SENSOR_BUDGET


        assert len(
            set(
                selected.tolist()
            )
        ) == SENSOR_BUDGET


        (
            next_state,
            reward,
            done,
            info,
        ) = env.step(
            action,
            nh4_weights,
            sno_weights,
        )


        # ----------------------------------------------------
        # Store transition
        # ----------------------------------------------------

        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )


        state = next_state


        episode_reward += reward

        episode_actions.append(
            action
        )

        episode_indices.append(
            current_index
        )

        all_training_rewards.append(
            reward
        )

        all_training_actions.append(
            action
        )

        all_training_indices.append(
            current_index
        )


        global_step += 1


        # ----------------------------------------------------
        # Learn after warm-up.
        # ----------------------------------------------------

        if len(
            replay_buffer
        ) >= WARMUP_STEPS:

            loss = optimize_model(
                policy_net,
                target_net,
                optimizer,
                replay_buffer
            )

            episode_losses_current.append(
                loss
            )

            # ------------------------------------------------
            # Target-network synchronization.
            # ------------------------------------------------

            if (
                global_step
                %
                TARGET_UPDATE_FREQUENCY
                ==
                0
            ):

                target_net.load_state_dict(
                    policy_net.state_dict()
                )


    # ========================================================
    # EPISODE SUMMARY
    # ========================================================

    episode_rewards.append(
        episode_reward
    )

    episode_losses.append(
        float(
            np.mean(
                episode_losses_current
            )
        )
        if episode_losses_current
        else np.nan
    )

    episode_action_counts.append(
        len(
            np.unique(
                episode_actions
            )
        )
    )

    episode_lengths.append(
        len(
            episode_actions
        )
    )


    print()
    print(
        f"[INFO] Episode "
        f"{episode + 1}/{TRAIN_EPISODES}"
    )

    print(
        f"       Steps: "
        f"{len(episode_actions)}"
    )

    print(
        f"       Reward: "
        f"{episode_reward:.6f}"
    )

    print(
        f"       Mean loss: "
        f"{episode_losses[-1]:.6f}"
    )

    print(
        f"       Unique actions: "
        f"{episode_action_counts[-1]}"
    )

    print(
        f"       Global step: "
        f"{global_step}"
    )

    print(
        f"       Epsilon: "
        f"{epsilon_by_step(global_step):.6f}"
    )


# ============================================================
# TRAINING INTEGRITY
# ============================================================

assert global_step == (
    TRAIN_EPISODES
    *
    (
        TRAIN_END
        -
        TRAIN_START
        +
        1
    )
)


assert len(
    replay_buffer
) == global_step


assert all(
    index <= TRAIN_END
    for index
    in all_training_indices
)


assert not any(
    index >= EVAL_START
    for index
    in all_training_indices
)


assert np.isfinite(
    np.asarray(
        all_training_rewards
    )
).all()


finite_losses = np.asarray(
    [
        loss
        for loss
        in episode_losses
        if np.isfinite(loss)
    ]
)


assert len(
    finite_losses
) > 0


assert np.isfinite(
    finite_losses
).all()


print()
print(
    "[PASS] Training remained entirely "
    "inside training indices."
)

print(
    "[PASS] No held-out evaluation index "
    "entered replay memory."
)

print(
    "[PASS] Training rewards are finite."
)

print(
    "[PASS] Training losses are finite."
)


# ============================================================
# ACTION CONSTRAINT
# ============================================================

all_training_actions_array = np.asarray(
    all_training_actions,
    dtype=np.int64
)


assert np.all(
    (
        all_training_actions_array
        >=
        0
    )
    &
    (
        all_training_actions_array
        <
        ACTION_DIM
    )
)


for action in np.unique(
    all_training_actions_array
):

    selected = ACTION_TABLE[
        action
    ]

    assert len(
        selected
    ) == SENSOR_BUDGET

    assert len(
        set(
            selected.tolist()
        )
    ) == SENSOR_BUDGET


print(
    "[PASS] Exactly-four constraint maintained "
    "throughout training."
)


# ============================================================
# HELD-OUT EVALUATION
# ============================================================

print()
print(
    "[INFO] Beginning held-out evaluation..."
)


eval_env = create_environment(
    EVAL_START,
    EVAL_END
)


state = eval_env.reset()

done = False


evaluation_rewards = []

evaluation_actions = []

evaluation_indices = []

evaluation_errors_nh4 = []

evaluation_errors_sno = []

evaluation_predictions_nh4 = []

evaluation_predictions_sno = []

evaluation_actual_nh4 = []

evaluation_actual_sno = []


while not done:

    current_index = (
        eval_env.current_index
    )


    assert (
        current_index
        >=
        EVAL_START
    )

    assert (
        current_index
        <=
        EVAL_END
    )


    # --------------------------------------------------------
    # Evaluation is greedy.
    # No random exploration.
    # --------------------------------------------------------

    action = select_action(
        policy_net,
        state,
        epsilon=0.0
    )


    assert (
        0 <= action < ACTION_DIM
    )


    selected = ACTION_TABLE[
        action
    ]


    assert len(
        selected
    ) == SENSOR_BUDGET


    (
        next_state,
        reward,
        done,
        info,
    ) = eval_env.step(
        action,
        nh4_weights,
        sno_weights,
    )


    evaluation_rewards.append(
        reward
    )

    evaluation_actions.append(
        action
    )

    evaluation_indices.append(
        current_index
    )

    evaluation_errors_nh4.append(
        info["error_nh4"]
    )

    evaluation_errors_sno.append(
        info["error_sno"]
    )

    evaluation_predictions_nh4.append(
        info["predicted_nh4"]
    )

    evaluation_predictions_sno.append(
        info["predicted_sno"]
    )

    evaluation_actual_nh4.append(
        info["actual_nh4"]
    )

    evaluation_actual_sno.append(
        info["actual_sno"]
    )


    state = next_state


# ============================================================
# EVALUATION INTEGRITY
# ============================================================

evaluation_indices = np.asarray(
    evaluation_indices,
    dtype=np.int64
)

evaluation_actions = np.asarray(
    evaluation_actions,
    dtype=np.int64
)

evaluation_rewards = np.asarray(
    evaluation_rewards,
    dtype=np.float64
)

evaluation_errors_nh4 = np.asarray(
    evaluation_errors_nh4,
    dtype=np.float64
)

evaluation_errors_sno = np.asarray(
    evaluation_errors_sno,
    dtype=np.float64
)

evaluation_predictions_nh4 = np.asarray(
    evaluation_predictions_nh4,
    dtype=np.float64
)

evaluation_predictions_sno = np.asarray(
    evaluation_predictions_sno,
    dtype=np.float64
)

evaluation_actual_nh4 = np.asarray(
    evaluation_actual_nh4,
    dtype=np.float64
)

evaluation_actual_sno = np.asarray(
    evaluation_actual_sno,
    dtype=np.float64
)


assert len(
    evaluation_indices
) == len(
    eval_indices
)


assert np.array_equal(
    evaluation_indices,
    eval_indices
)


assert np.all(
    evaluation_indices >= EVAL_START
)

assert np.all(
    evaluation_indices <= EVAL_END
)


assert np.isfinite(
    evaluation_rewards
).all()

assert np.isfinite(
    evaluation_errors_nh4
).all()

assert np.isfinite(
    evaluation_errors_sno
).all()

assert np.isfinite(
    evaluation_predictions_nh4
).all()

assert np.isfinite(
    evaluation_predictions_sno
).all()

assert np.isfinite(
    evaluation_actual_nh4
).all()

assert np.isfinite(
    evaluation_actual_sno
).all()


print(
    "[PASS] Held-out evaluation used only "
    "evaluation indices."
)


# ============================================================
# EVALUATION ACTION CONSTRAINT
# ============================================================

for action in np.unique(
    evaluation_actions
):

    selected = ACTION_TABLE[
        action
    ]

    assert len(
        selected
    ) == SENSOR_BUDGET

    assert len(
        set(
            selected.tolist()
        )
    ) == SENSOR_BUDGET


print(
    "[PASS] Exactly-four constraint maintained "
    "during evaluation."
)


# ============================================================
# METRICS
# ============================================================

rmse_nh4 = float(
    np.sqrt(
        np.mean(
            evaluation_errors_nh4 ** 2
        )
    )
)

rmse_sno = float(
    np.sqrt(
        np.mean(
            evaluation_errors_sno ** 2
        )
    )
)


baseline_nh4 = float(
    np.sqrt(
        np.mean(
            evaluation_actual_nh4 ** 2
        )
    )
)

baseline_sno = float(
    np.sqrt(
        np.mean(
            evaluation_actual_sno ** 2
        )
    )
)


improvement_nh4 = (
    1.0
    -
    rmse_nh4
    /
    baseline_nh4
) * 100.0


improvement_sno = (
    1.0
    -
    rmse_sno
    /
    baseline_sno
) * 100.0


mean_reward = float(
    np.mean(
        evaluation_rewards
    )
)


unique_eval_actions = int(
    len(
        np.unique(
            evaluation_actions
        )
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print(
    "------------------------------------------------------------"
)

print(
    "HELD-OUT EVALUATION RESULTS"
)

print(
    "------------------------------------------------------------"
)

print(
    f"[RESULT] Evaluation decisions: "
    f"{len(evaluation_indices)}"
)

print(
    f"[RESULT] Unique actions selected: "
    f"{unique_eval_actions}"
)

print(
    f"[RESULT] Mean reward: "
    f"{mean_reward:.6f}"
)

print()
print(
    f"[RESULT] ΔNH4 RMSE: "
    f"{rmse_nh4:.6f}"
)

print(
    f"[RESULT] Zero-change ΔNH4 baseline: "
    f"{baseline_nh4:.6f}"
)

print(
    f"[RESULT] Relative ΔNH4 improvement: "
    f"{improvement_nh4:.2f}%"
)

print()
print(
    f"[RESULT] ΔSNO RMSE: "
    f"{rmse_sno:.6f}"
)

print(
    f"[RESULT] Zero-change ΔSNO baseline: "
    f"{baseline_sno:.6f}"
)

print(
    f"[RESULT] Relative ΔSNO improvement: "
    f"{improvement_sno:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_ca_rdqn",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


checkpoint_path = os.path.join(
    OUTPUT_DIR,
    f"ca_rdqn_seed_{SEED}.pt"
)


results_path = os.path.join(
    OUTPUT_DIR,
    f"ca_rdqn_seed_{SEED}_results.json"
)


history_path = os.path.join(
    OUTPUT_DIR,
    f"ca_rdqn_seed_{SEED}_history.npz"
)


# ------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------

torch.save(
    {
        "seed": SEED,
        "policy_state_dict":
            policy_net.state_dict(),
        "target_state_dict":
            target_net.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "global_step": global_step,
        "config": {
            "gamma": GAMMA,
            "learning_rate":
                LEARNING_RATE,
            "batch_size":
                BATCH_SIZE,
            "replay_capacity":
                REPLAY_CAPACITY,
            "warmup_steps":
                WARMUP_STEPS,
            "target_update":
                TARGET_UPDATE_FREQUENCY,
            "epsilon_start":
                EPSILON_START,
            "epsilon_end":
                EPSILON_END,
            "epsilon_decay":
                EPSILON_DECAY_STEPS,
            "episodes":
                TRAIN_EPISODES,
        },
    },
    checkpoint_path
)


# ------------------------------------------------------------
# Results JSON
# ------------------------------------------------------------

results = {
    "seed": SEED,
    "training_start": TRAIN_START,
    "training_end": TRAIN_END,
    "evaluation_start": EVAL_START,
    "evaluation_end": EVAL_END,
    "training_episodes": TRAIN_EPISODES,
    "training_steps": global_step,
    "evaluation_steps":
        int(len(evaluation_indices)),
    "unique_evaluation_actions":
        unique_eval_actions,
    "mean_evaluation_reward":
        mean_reward,
    "nh4_rmse":
        rmse_nh4,
    "sno_rmse":
        rmse_sno,
    "nh4_zero_change_baseline":
        baseline_nh4,
    "sno_zero_change_baseline":
        baseline_sno,
    "nh4_relative_improvement_percent":
        improvement_nh4,
    "sno_relative_improvement_percent":
        improvement_sno,
}


with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ------------------------------------------------------------
# Training/evaluation history
# ------------------------------------------------------------

np.savez(
    history_path,
    episode_rewards=np.asarray(
        episode_rewards,
        dtype=np.float64
    ),
    episode_losses=np.asarray(
        episode_losses,
        dtype=np.float64
    ),
    episode_action_counts=np.asarray(
        episode_action_counts,
        dtype=np.int64
    ),
    episode_lengths=np.asarray(
        episode_lengths,
        dtype=np.int64
    ),
    evaluation_indices=
        evaluation_indices,
    evaluation_actions=
        evaluation_actions,
    evaluation_rewards=
        evaluation_rewards,
    evaluation_errors_nh4=
        evaluation_errors_nh4,
    evaluation_errors_sno=
        evaluation_errors_sno,
    evaluation_predictions_nh4=
        evaluation_predictions_nh4,
    evaluation_predictions_sno=
        evaluation_predictions_sno,
    evaluation_actual_nh4=
        evaluation_actual_nh4,
    evaluation_actual_sno=
        evaluation_actual_sno,
)


assert os.path.exists(
    checkpoint_path
)

assert os.path.exists(
    results_path
)

assert os.path.exists(
    history_path
)


print()
print(
    "[PASS] Final model checkpoint saved."
)

print(
    "[PASS] Evaluation results saved."
)

print(
    "[PASS] Training/evaluation history saved."
)

print(
    f"[INFO] Output directory: "
    f"{OUTPUT_DIR}"
)


# ============================================================
# FINAL
# ============================================================

print()
print(
    "[PASS] CA-RDQN single-seed training completed."
)

print(
    "[PASS] Held-out evaluation completed."
)

print(
    "[PASS] No evaluation samples entered training."
)

print(
    "[PASS] Four-sensor constraint maintained."
)

print(
    "[PASS] Results and checkpoint preserved."
)

print("=" * 70)
print(
    "STEP 68 COMPLETED"
)
print("=" * 70)