# ============================================================
# FINAL CA-DQN ABLATION
# ============================================================
#
# Article:
# Constraint-Aware Deep Q-Network for Dynamic Sensor Activation
# Scheduling in Wastewater Treatment Plants:
# A Multi-Seed Robustness and Ablation Study
#
# ABLATION:
# CA-DQN = Constraint-Aware DQN without recurrent temporal memory
#
# SCIENTIFIC PURPOSE
# ------------------
# This experiment isolates the contribution of recurrent
# temporal representation.
#
# MAIN MODEL:
#     4-step observation sequence
#         -> GRU(128)
#         -> FC(64)
#         -> 495 Q-values
#
# ABLATION:
#     current 24-D observation
#         -> FC(128)
#         -> FC(64)
#         -> 495 Q-values
#
# All other experimental components remain unchanged.
#
# ============================================================

import argparse
import json
import os
import random
from builtins import print

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Final CA-DQN no-recurrence ablation."
)

parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Independent random seed."
)

args = parser.parse_args()

SEED = args.seed


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in os.sys.path:
    os.sys.path.insert(
        0,
        SRC_DIR
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from dqn_data_loader import (
    train_indices,
    eval_indices,
    HISTORY,
    SENSOR_COUNT,
)

from ca_rdqn_training_environment import (
    ACTION_TABLE,
    ACTION_DIM,
    OBSERVATION_DIM,
    CausalRDQNEnvironment,
)

nh4_weights = (
    CausalRDQNEnvironment._load_weights(
        "causal_estimator_nh4_weights.npz"
    )
)

sno_weights = (
    CausalRDQNEnvironment._load_weights(
        "causal_estimator_sno_weights.npz"
    )
)

# ============================================================
# LOCKED EXPERIMENTAL CONFIGURATION
# ============================================================

TRAIN_START = 4
TRAIN_END = 862

EVAL_START = 863
EVAL_END = 1340

TRAIN_EPISODES = 24

DECISION_MINUTES = 15
FORECAST_MINUTES = 30

GAMMA = 0.99

LEARNING_RATE = 0.001

BATCH_SIZE = 64

REPLAY_CAPACITY = 50000

WARMUP_STEPS = 64

TARGET_UPDATE_FREQUENCY = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 20000

GRADIENT_CLIP = 1.0

HIDDEN_DIM = 128
FC_DIM = 64


# ============================================================
# CONSTRAINT DEFINITION
# ============================================================
#
# The production environment defines the feasible action
# space as all combinations of 4 sensors from 12 sensors:
#
#     C(12,4) = 495
#
# SENSOR_BUDGET is intentionally defined locally here rather
# than modifying the production environment.
# ============================================================

SENSOR_BUDGET = 4


# ============================================================
# PROTOCOL ASSERTIONS
# ============================================================

assert SENSOR_COUNT == 12

assert SENSOR_BUDGET == 4

assert ACTION_DIM == 495

assert OBSERVATION_DIM == 24

assert HISTORY == 4

assert len(ACTION_TABLE) == 495

assert TRAIN_START == 4

assert TRAIN_END == 862

assert EVAL_START == 863

assert EVAL_END == 1340

assert TRAIN_EPISODES == 24

assert DECISION_MINUTES == 15

assert FORECAST_MINUTES == 30

assert GAMMA == 0.99

assert LEARNING_RATE == 0.001

assert BATCH_SIZE == 64

assert REPLAY_CAPACITY == 50000

assert WARMUP_STEPS == 64

assert TARGET_UPDATE_FREQUENCY == 500

assert EPSILON_START == 1.0

assert EPSILON_END == 0.05

assert EPSILON_DECAY == 20000


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_ca_dqn_ablation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    f"ca_dqn_ablation_seed_{SEED}.pt"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    f"ca_dqn_ablation_seed_{SEED}_results.json"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    f"ca_dqn_ablation_seed_{SEED}_history.npz"
)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity
    ):

        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
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

        if len(self.buffer) < self.capacity:

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
            replace=False
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
            dones
        )

    def __len__(
        self
    ):

        return len(
            self.buffer
        )


# ============================================================
# CA-DQN NETWORK
# ============================================================
#
# NO GRU.
#
# The network receives ONLY the current 24-dimensional
# observation:
#
#     24 -> 128 -> 64 -> 495
#
# This is the architectural ablation of the recurrent
# CA-RDQN model.
# ============================================================

class CADQN(
    nn.Module
):

    def __init__(
        self,
        observation_dim=OBSERVATION_DIM,
        hidden_dim=HIDDEN_DIM,
        fc_dim=FC_DIM,
        action_dim=ACTION_DIM
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            observation_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            fc_dim
        )

        self.output = nn.Linear(
            fc_dim,
            action_dim
        )

    def forward(
        self,
        x
    ):

        assert x.ndim == 2

        assert (
            x.shape[1]
            == OBSERVATION_DIM
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        return self.output(x)


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model
):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# EPSILON SCHEDULE
# ============================================================

def epsilon_by_step(
    step
):

    fraction = min(
        step / EPSILON_DECAY,
        1.0
    )

    return (
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
# CURRENT OBSERVATION EXTRACTION
# ============================================================

def current_observation(
    state
):

    state = np.asarray(
        state,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # The production environment returns the 4-step state
    # sequence. For this ablation, ONLY the latest row is
    # supplied to the feed-forward network.
    # --------------------------------------------------------

    if state.ndim == 2:

        assert state.shape == (
            HISTORY,
            OBSERVATION_DIM
        )

        state = state[-1]

    assert state.ndim == 1

    assert state.shape == (
        OBSERVATION_DIM,
    )

    assert np.all(
        np.isfinite(state)
    )

    return state


# ============================================================
# ACTION SELECTION
# ============================================================

def select_action(
    network,
    state,
    epsilon
):

    if random.random() < epsilon:

        return random.randrange(
            ACTION_DIM
        )

    state = current_observation(
        state
    )

    state_tensor = torch.from_numpy(
        state
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = network(
            state_tensor
        )

    assert q_values.shape == (
        1,
        ACTION_DIM
    )

    return int(
        torch.argmax(
            q_values,
            dim=1
        ).item()
    )


# ============================================================
# DOUBLE-DQN OPTIMIZATION
# ============================================================

def optimize_model(
    policy_net,
    target_net,
    optimizer,
    replay_buffer
):

    (
        states,
        actions,
        rewards,
        next_states,
        dones
    ) = replay_buffer.sample(
        BATCH_SIZE
    )

    states_tensor = torch.from_numpy(
        states
    )

    actions_tensor = torch.from_numpy(
        actions
    ).long()

    rewards_tensor = torch.from_numpy(
        rewards
    )

    next_states_tensor = torch.from_numpy(
        next_states
    )

    dones_tensor = torch.from_numpy(
        dones
    )

    # --------------------------------------------------------
    # Current Q
    # --------------------------------------------------------

    q_values = policy_net(
        states_tensor
    )

    current_q = q_values.gather(
        1,
        actions_tensor.unsqueeze(1)
    ).squeeze(1)

    # --------------------------------------------------------
    # Double-DQN target
    # --------------------------------------------------------

    with torch.no_grad():

        next_policy_q = policy_net(
            next_states_tensor
        )

        next_actions = torch.argmax(
            next_policy_q,
            dim=1
        )

        next_target_q = target_net(
            next_states_tensor
        )

        next_q = next_target_q.gather(
            1,
            next_actions.unsqueeze(1)
        ).squeeze(1)

        target = (
            rewards_tensor
            +
            GAMMA
            *
            next_q
            *
            (
                1.0
                -
                dones_tensor
            )
        )

    # --------------------------------------------------------
    # Huber loss
    # --------------------------------------------------------

    loss = nn.SmoothL1Loss()(
        current_q,
        target
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        GRADIENT_CLIP
    )

    optimizer.step()

    return float(
        loss.item()
    )


# ============================================================
# STARTUP
# ============================================================

print()
print("=" * 70)
print(
    f"FINAL CA-DQN ABLATION — SEED {SEED}"
)
print("=" * 70)

print()
print(
    "[INFO] Ablation: NO RECURRENCE."
)

print(
    "[INFO] Current 24-D observation only."
)

print(
    f"[INFO] Candidate sensors: {SENSOR_COUNT}"
)

print(
    f"[INFO] Active sensors: {SENSOR_BUDGET}"
)

print(
    f"[INFO] Feasible actions: {ACTION_DIM}"
)

print(
    f"[INFO] Environment history: {HISTORY}"
)

print(
    "[INFO] Network history input: 1 current observation"
)

print(
    f"[INFO] Network: "
    f"{OBSERVATION_DIM} -> "
    f"{HIDDEN_DIM} -> "
    f"{FC_DIM} -> "
    f"{ACTION_DIM}"
)

print(
    f"[INFO] Training episodes: {TRAIN_EPISODES}"
)

print(
    f"[INFO] Training indices: "
    f"{TRAIN_START}–{TRAIN_END}"
)

print(
    f"[INFO] Evaluation indices: "
    f"{EVAL_START}–{EVAL_END}"
)


# ============================================================
# NETWORKS
# ============================================================

policy_net = CADQN()

target_net = CADQN()

target_net.load_state_dict(
    policy_net.state_dict()
)

target_net.eval()

parameter_count = count_parameters(
    policy_net
)

print()
print(
    "[PASS] CA-DQN policy network initialized."
)

print(
    "[PASS] CA-DQN target network initialized."
)

print(
    f"[INFO] Trainable parameters: "
    f"{parameter_count}"
)

# No GRU must exist.

assert not hasattr(
    policy_net,
    "gru"
)

print(
    "[PASS] No recurrent GRU present."
)


# ============================================================
# FORWARD-PASS VALIDATION
# ============================================================

test_state = np.zeros(
    OBSERVATION_DIM,
    dtype=np.float32
)

test_tensor = torch.from_numpy(
    test_state
).unsqueeze(0)

with torch.no_grad():

    test_q = policy_net(
        test_tensor
    )

assert test_q.shape == (
    1,
    ACTION_DIM
)

assert np.all(
    np.isfinite(
        test_q.numpy()
    )
)

print(
    "[PASS] Current-observation forward pass verified."
)

print(
    f"[PASS] Q-value shape: "
    f"{tuple(test_q.shape)}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE
)

print(
    "[PASS] Adam optimizer initialized."
)


# ============================================================
# REPLAY BUFFER
# ============================================================

replay_buffer = ReplayBuffer(
    REPLAY_CAPACITY
)

print(
    "[PASS] Replay buffer initialized."
)


# ============================================================
# TRAINING HISTORY
# ============================================================

episode_rewards = []

episode_losses = []

episode_action_counts = []

episode_lengths = []

all_training_rewards = []

all_training_actions = []

all_training_indices = []

global_step = 0


# ============================================================
# TRAINING
# ============================================================

for episode in range(
    TRAIN_EPISODES
):

    env = CausalRDQNEnvironment(
        TRAIN_START,
        TRAIN_END,
    )

    raw_state = env.reset()

    state = current_observation(
        raw_state
    )

    assert state.shape == (
        OBSERVATION_DIM,
    )

    done = False

    episode_reward = 0.0

    episode_losses_current = []

    episode_actions = []

    episode_indices = []


    # ========================================================
    # EPISODE LOOP
    # ========================================================

    while not done:

        current_index = (
            env.current_index
        )

        assert (
            TRAIN_START
            <= current_index
            <= TRAIN_END
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
            0
            <= action
            <
            ACTION_DIM
        )

        # ----------------------------------------------------
        # Feasible action verification
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Environment transition
        # ----------------------------------------------------

        (
            next_raw_state,
            reward,
            done,
            info
        ) = env.step(
            action,
            nh4_weights,
            sno_weights
        )

        # ----------------------------------------------------
        # Extract current observation only.
        # ----------------------------------------------------

        next_state = current_observation(
            next_raw_state
        )

        assert next_state.shape == (
            OBSERVATION_DIM,
        )

        assert np.all(
            np.isfinite(
                next_state
            )
        )

        assert np.isfinite(
            reward
        )

        # ----------------------------------------------------
        # Verify actual environment outputs.
        # ----------------------------------------------------

        assert np.isfinite(
            info["predicted_nh4"]
        )

        assert np.isfinite(
            info["predicted_sno"]
        )

        assert np.isfinite(
            info["actual_nh4"]
        )

        assert np.isfinite(
            info["actual_sno"]
        )

        assert np.isfinite(
            info["error_nh4"]
        )

        assert np.isfinite(
            info["error_sno"]
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

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

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
        # Optimization
        # ----------------------------------------------------

        if (
            len(replay_buffer)
            >= WARMUP_STEPS
        ):

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
            # Target network update
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
        f"       Unique actions: "
        f"{len(np.unique(episode_actions))}"
    )

    print(
        f"       Epsilon: "
        f"{epsilon:.6f}"
    )

    if episode_losses_current:

        print(
            f"       Mean loss: "
            f"{np.mean(episode_losses_current):.6f}"
        )

    else:

        print(
            "       Mean loss: warm-up"
        )


# ============================================================
# TRAINING COMPLETION CHECK
# ============================================================

expected_training_steps = (
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

assert global_step == (
    expected_training_steps
)

print()
print(
    "[PASS] Training transition count verified."
)

print(
    f"[PASS] Training transitions: "
    f"{global_step}"
)


# ============================================================
# FINAL TARGET SYNCHRONIZATION
# ============================================================

target_net.load_state_dict(
    policy_net.state_dict()
)

print(
    "[PASS] Final target-network synchronization completed."
)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

checkpoint = {

    "seed":
        SEED,

    "model_state_dict":
        policy_net.state_dict(),

    "target_state_dict":
        target_net.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "global_step":
        global_step,

    "architecture":
        "CA-DQN",

    "ablation":
        "no_recurrence",

    "recurrent":
        False,

    "observation_dimension":
        OBSERVATION_DIM,

    "network_history":
        1,

    "environment_history":
        HISTORY,

    "action_dimension":
        ACTION_DIM,

    "sensor_count":
        SENSOR_COUNT,

    "sensor_budget":
        SENSOR_BUDGET,

    "training_start":
        TRAIN_START,

    "training_end":
        TRAIN_END,

    "evaluation_start":
        EVAL_START,

    "evaluation_end":
        EVAL_END,

    "training_episodes":
        TRAIN_EPISODES,

    "decision_minutes":
        DECISION_MINUTES,

    "forecast_minutes":
        FORECAST_MINUTES,

    "gamma":
        GAMMA,

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
        EPSILON_DECAY,

    "parameter_count":
        parameter_count,
}

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

assert os.path.exists(
    CHECKPOINT_PATH
)

print(
    "[PASS] CA-DQN checkpoint saved."
)


# ============================================================
# HELD-OUT EVALUATION
# ============================================================

eval_env = CausalRDQNEnvironment(
    EVAL_START,
    EVAL_END,
)

raw_eval_state = eval_env.reset()

eval_state = current_observation(
    raw_eval_state
)

assert eval_state.shape == (
    OBSERVATION_DIM,
)

evaluation_actions = []

evaluation_rewards = []

evaluation_indices = []

evaluation_targets_nh4 = []

evaluation_predictions_nh4 = []

evaluation_targets_sno = []

evaluation_predictions_sno = []

evaluation_errors_nh4 = []

evaluation_errors_sno = []

done = False

policy_net.eval()


with torch.no_grad():

    while not done:

        current_index = (
            eval_env.current_index
        )

        assert (
            EVAL_START
            <= current_index
            <= EVAL_END
        )

        state_tensor = torch.from_numpy(
            eval_state
        ).unsqueeze(0)

        q_values = policy_net(
            state_tensor
        )

        assert q_values.shape == (
            1,
            ACTION_DIM
        )

        action = int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
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
            next_raw_state,
            reward,
            done,
            info
        ) = eval_env.step(
            action,
            nh4_weights,
            sno_weights
        )

        # ----------------------------------------------------
        # Store actual environment outputs.
        # ----------------------------------------------------

        evaluation_actions.append(
            action
        )

        evaluation_rewards.append(
            float(reward)
        )

        evaluation_indices.append(
            current_index
        )

        evaluation_targets_nh4.append(
            float(
                info["actual_nh4"]
            )
        )

        evaluation_predictions_nh4.append(
            float(
                info["predicted_nh4"]
            )
        )

        evaluation_targets_sno.append(
            float(
                info["actual_sno"]
            )
        )

        evaluation_predictions_sno.append(
            float(
                info["predicted_sno"]
            )
        )

        evaluation_errors_nh4.append(
            float(
                info["error_nh4"]
            )
        )

        evaluation_errors_sno.append(
            float(
                info["error_sno"]
            )
        )

        # ----------------------------------------------------
        # Numerical checks
        # ----------------------------------------------------

        assert np.isfinite(
            reward
        )

        assert np.isfinite(
            info["actual_nh4"]
        )

        assert np.isfinite(
            info["predicted_nh4"]
        )

        assert np.isfinite(
            info["actual_sno"]
        )

        assert np.isfinite(
            info["predicted_sno"]
        )

        # ----------------------------------------------------
        # Current observation only
        # ----------------------------------------------------

        eval_state = current_observation(
            next_raw_state
        )


# ============================================================
# EVALUATION AUDIT
# ============================================================

assert len(
    evaluation_actions
) == 478

assert len(
    evaluation_indices
) == 478

assert evaluation_indices[0] >= (
    EVAL_START
)

assert evaluation_indices[-1] <= (
    EVAL_END
)

for action in evaluation_actions:

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


print()
print(
    "[PASS] Held-out evaluation completed."
)

print(
    f"[PASS] Evaluation steps: "
    f"{len(evaluation_actions)}"
)

print(
    "[PASS] Every evaluation action contains exactly four sensors."
)


# ============================================================
# METRICS
# ============================================================

nh4_targets = np.asarray(
    evaluation_targets_nh4,
    dtype=np.float64
)

nh4_predictions = np.asarray(
    evaluation_predictions_nh4,
    dtype=np.float64
)

sno_targets = np.asarray(
    evaluation_targets_sno,
    dtype=np.float64
)

sno_predictions = np.asarray(
    evaluation_predictions_sno,
    dtype=np.float64
)

nh4_errors = (
    nh4_predictions
    -
    nh4_targets
)

sno_errors = (
    sno_predictions
    -
    sno_targets
)

nh4_rmse = float(
    np.sqrt(
        np.mean(
            nh4_errors ** 2
        )
    )
)

sno_rmse = float(
    np.sqrt(
        np.mean(
            sno_errors ** 2
        )
    )
)

nh4_mae = float(
    np.mean(
        np.abs(
            nh4_errors
        )
    )
)

sno_mae = float(
    np.mean(
        np.abs(
            sno_errors
        )
    )

)

mean_evaluation_reward = float(
    np.mean(
        evaluation_rewards
    )
)

unique_evaluation_actions = len(
    np.unique(
        evaluation_actions
    )
)


# ============================================================
# ZERO-CHANGE BASELINES
# ============================================================
#
# These are the established held-out zero-change baselines
# associated with the 478-step evaluation interval.
# ============================================================

NH4_ZERO_CHANGE_BASELINE = 0.5311878570546589

SNO_ZERO_CHANGE_BASELINE = 0.2767668238145015


nh4_relative_improvement = (
    (
        NH4_ZERO_CHANGE_BASELINE
        -
        nh4_rmse
    )
    /
    NH4_ZERO_CHANGE_BASELINE
    *
    100.0
)

sno_relative_improvement = (
    (
        SNO_ZERO_CHANGE_BASELINE
        -
        sno_rmse
    )
    /
    SNO_ZERO_CHANGE_BASELINE
    *
    100.0
)


# ============================================================
# RESULTS
# ============================================================

results = {

    "seed":
        SEED,

    "architecture":
        "CA-DQN",

    "ablation":
        "no_recurrence",

    "recurrent":
        False,

    "observation_dimension":
        OBSERVATION_DIM,

    "network_history":
        1,

    "environment_history":
        HISTORY,

    "sensor_count":
        SENSOR_COUNT,

    "sensor_budget":
        SENSOR_BUDGET,

    "action_dimension":
        ACTION_DIM,

    "training_start":
        TRAIN_START,

    "training_end":
        TRAIN_END,

    "evaluation_start":
        EVAL_START,

    "evaluation_end":
        EVAL_END,

    "training_episodes":
        TRAIN_EPISODES,

    "training_steps":
        global_step,

    "evaluation_steps":
        len(evaluation_actions),

    "unique_evaluation_actions":
        unique_evaluation_actions,

    "mean_evaluation_reward":
        mean_evaluation_reward,

    "nh4_rmse":
        nh4_rmse,

    "nh4_mae":
        nh4_mae,

    "sno_rmse":
        sno_rmse,

    "sno_mae":
        sno_mae,

    "nh4_zero_change_baseline":
        NH4_ZERO_CHANGE_BASELINE,

    "sno_zero_change_baseline":
        SNO_ZERO_CHANGE_BASELINE,

    "nh4_relative_improvement_percent":
        float(
            nh4_relative_improvement
        ),

    "sno_relative_improvement_percent":
        float(
            sno_relative_improvement
        ),

    "gamma":
        GAMMA,

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
        EPSILON_DECAY,

    "decision_minutes":
        DECISION_MINUTES,

    "forecast_minutes":
        FORECAST_MINUTES,

    "parameter_count":
        parameter_count,
}


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=2
    )

assert os.path.exists(
    RESULTS_PATH
)

print(
    "[PASS] CA-DQN results saved."
)


# ============================================================
# SAVE HISTORY
# ============================================================

np.savez(
    HISTORY_PATH,

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

    all_training_rewards=np.asarray(
        all_training_rewards,
        dtype=np.float64
    ),

    all_training_actions=np.asarray(
        all_training_actions,
        dtype=np.int64
    ),

    all_training_indices=np.asarray(
        all_training_indices,
        dtype=np.int64
    ),

    evaluation_actions=np.asarray(
        evaluation_actions,
        dtype=np.int64
    ),

    evaluation_rewards=np.asarray(
        evaluation_rewards,
        dtype=np.float64
    ),

    evaluation_indices=np.asarray(
        evaluation_indices,
        dtype=np.int64
    ),

    evaluation_targets_nh4=nh4_targets,

    evaluation_predictions_nh4=nh4_predictions,

    evaluation_targets_sno=sno_targets,

    evaluation_predictions_sno=sno_predictions,

    evaluation_errors_nh4=nh4_errors,

    evaluation_errors_sno=sno_errors,
)

assert os.path.exists(
    HISTORY_PATH
)

print(
    "[PASS] Training/evaluation history saved."
)


# ============================================================
# FINAL INTEGRITY CHECKS
# ============================================================

assert global_step == 20616

assert len(
    evaluation_actions
) == 478

assert len(
    evaluation_rewards
) == 478

assert np.all(
    np.isfinite(
        evaluation_rewards
    )
)

assert np.isfinite(
    nh4_rmse
)

assert np.isfinite(
    sno_rmse
)

assert np.isfinite(
    nh4_relative_improvement
)

assert np.isfinite(
    sno_relative_improvement
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)

print(
    "STEP — FINAL CA-DQN NO-RECURRENCE ABLATION"
)

print("=" * 70)

print(
    f"[PASS] Seed: {SEED}"
)

print(
    "[PASS] Recurrent layer: NONE"
)

print(
    f"[PASS] Current observation dimension: "
    f"{OBSERVATION_DIM}"
)

print(
    f"[PASS] Feasible actions: "
    f"{ACTION_DIM}"
)

print(
    f"[PASS] Active sensors per action: "
    f"{SENSOR_BUDGET}"
)

print(
    f"[PASS] Training transitions: "
    f"{global_step}"
)

print(
    f"[PASS] Evaluation steps: "
    f"{len(evaluation_actions)}"
)

print(
    f"[PASS] Unique evaluation actions: "
    f"{unique_evaluation_actions}"
)

print(
    f"[RESULT] Mean evaluation reward: "
    f"{mean_evaluation_reward:.6f}"
)

print(
    f"[RESULT] ΔNH4 RMSE: "
    f"{nh4_rmse:.6f}"
)

print(
    f"[RESULT] ΔNH4 relative improvement: "
    f"{nh4_relative_improvement:.2f}%"
)

print(
    f"[RESULT] ΔSNO RMSE: "
    f"{sno_rmse:.6f}"
)

print(
    f"[RESULT] ΔSNO relative improvement: "
    f"{sno_relative_improvement:.2f}%"
)

print(
    f"[PASS] Checkpoint: "
    f"{CHECKPOINT_PATH}"
)

print(
    f"[PASS] Results: "
    f"{RESULTS_PATH}"
)

print(
    f"[PASS] History: "
    f"{HISTORY_PATH}"
)

print("=" * 70)

print(
    "STEP COMPLETED"
)

print("=" * 70)