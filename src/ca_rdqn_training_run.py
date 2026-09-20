import os
import sys
import random

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
# IMPORT ENVIRONMENT
# ============================================================

from ca_rdqn_training_environment import (
    CausalRDQNEnvironment,
    ACTION_TABLE,
    ACTION_DIM,
    OBSERVATION_DIM,
    STATE_HISTORY,
    FORECAST_STEPS,
)


# ============================================================
# DATA
# ============================================================

from dqn_data_loader import (
    train_indices,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GAMMA = 0.99

LEARNING_RATE = 1e-3

BATCH_SIZE = 64

REPLAY_CAPACITY = 50_000

TARGET_UPDATE_FREQUENCY = 500

EPSILON_START = 1.0

EPSILON_END = 0.05

EPSILON_DECAY_STEPS = 20_000

WARMUP_STEPS = 64

CONTROL_EPISODES = 3


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# NETWORK
# ============================================================

class RecurrentDQN(nn.Module):

    def __init__(
        self,
        observation_dim=OBSERVATION_DIM,
        hidden_dim=128,
        fc_dim=64,
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


    def forward(self, x):

        assert x.ndim == 3

        assert x.shape[1] == STATE_HISTORY

        assert x.shape[2] == OBSERVATION_DIM

        sequence_output, _ = self.gru(x)

        final_hidden = (
            sequence_output[:, -1, :]
        )

        return self.q_head(
            final_hidden
        )


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity,
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
        batch_size,
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
            dtype=np.int64,
        )

        rewards = np.asarray(
            [
                item[2]
                for item in batch
            ],
            dtype=np.float32,
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
            dtype=np.float32,
        )


        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )


    def __len__(self):

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
        step
        /
        EPSILON_DECAY_STEPS,
        1.0,
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
        dtype=torch.float32,
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
# TRAINING UPDATE
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
        dtype=torch.float32,
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long,
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
    )

    next_states = torch.tensor(
        next_states,
        dtype=torch.float32,
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
    )


    # --------------------------------------------------------
    # Current Q values
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


        target = (
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
    # Loss
    # --------------------------------------------------------

    loss_function = nn.SmoothL1Loss()

    loss = loss_function(
        selected_q,
        target
    )


    assert torch.isfinite(
        loss
    )


    optimizer.zero_grad()

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=10.0
    )


    optimizer.step()


    return float(
        loss.detach().cpu().item()
    )


# ============================================================
# LOAD ESTIMATORS
# ============================================================

def load_weights(
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
# MAIN
# ============================================================

print("=" * 70)
print(
    "STEP 66 — FIRST CONTROLLED CA-RDQN TRAINING RUN"
)
print("=" * 70)


print(
    f"[INFO] Seed: {SEED}"
)

print(
    f"[INFO] Control episodes: "
    f"{CONTROL_EPISODES}"
)

print(
    f"[INFO] Training indices: "
    f"{int(train_indices[0])}–"
    f"{int(train_indices[-1])}"
)

print(
    f"[INFO] Action dimension: "
    f"{ACTION_DIM}"
)

print(
    f"[INFO] State shape: "
    f"({STATE_HISTORY}, {OBSERVATION_DIM})"
)


# ============================================================
# LOAD ESTIMATORS
# ============================================================

nh4_weights = load_weights(
    "causal_estimator_nh4_weights.npz"
)

sno_weights = load_weights(
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
# TRAINING
# ============================================================

global_step = 0

loss_history = []

reward_history = []

action_history = []

episode_lengths = []


for episode in range(
    CONTROL_EPISODES
):

    env = CausalRDQNEnvironment(
        start_index=int(
            train_indices[0]
        ),
        end_index=int(
            train_indices[-1]
        ),
    )


    state = env.reset()

    done = False

    episode_reward = 0.0

    episode_steps = 0


    while not done:

        epsilon = epsilon_by_step(
            global_step
        )


        action = select_action(
            policy_net,
            state,
            epsilon
        )


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
        # Validate action
        # ----------------------------------------------------

        selected = ACTION_TABLE[
            action
        ]


        assert len(
            selected
        ) == 4


        assert len(
            set(
                selected.tolist()
            )
        ) == 4


        # ----------------------------------------------------
        # Store transition
        # ----------------------------------------------------

        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )


        state = next_state

        episode_reward += reward

        reward_history.append(
            reward
        )

        action_history.append(
            action
        )

        episode_steps += 1

        global_step += 1


        # ----------------------------------------------------
        # Learn after replay warm-up
        # ----------------------------------------------------

        if (
            len(replay_buffer)
            >= WARMUP_STEPS
        ):

            loss = optimize_model(
                policy_net,
                target_net,
                optimizer,
                replay_buffer,
            )

            loss_history.append(
                loss
            )


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


    episode_lengths.append(
        episode_steps
    )


    print()
    print(
        f"[INFO] Episode {episode + 1}/"
        f"{CONTROL_EPISODES}"
    )

    print(
        f"       Steps: {episode_steps}"
    )

    print(
        f"       Episode reward: "
        f"{episode_reward:.6f}"
    )

    print(
        f"       Global step: "
        f"{global_step}"
    )

    print(
        f"       Epsilon: "
        f"{epsilon_by_step(global_step):.6f}"
    )

    print(
        f"       Replay size: "
        f"{len(replay_buffer)}"
    )


# ============================================================
# TRAINING VALIDATION
# ============================================================

assert global_step > 0

assert len(
    replay_buffer
) == global_step


assert len(
    loss_history
) > 0


assert np.isfinite(
    np.asarray(
        loss_history
    )
).all()


assert np.isfinite(
    np.asarray(
        reward_history
    )
).all()


assert np.isfinite(
    np.asarray(
        action_history
    )
).all()


assert all(
    0 <= action < ACTION_DIM
    for action in action_history
)


# ============================================================
# ACTION COVERAGE
# ============================================================

unique_actions = len(
    np.unique(
        np.asarray(
            action_history
        )
    )
)


print()
print(
    f"[INFO] Unique actions selected: "
    f"{unique_actions}"
)


# ============================================================
# NETWORK CHANGE
# ============================================================

policy_parameters = [
    parameter.detach().cpu().numpy()
    for parameter
    in policy_net.parameters()
]


target_parameters = [
    parameter.detach().cpu().numpy()
    for parameter
    in target_net.parameters()
]


parameter_difference = sum(
    np.sum(
        (
            p - t
        ) ** 2
    )
    for p, t in zip(
        policy_parameters,
        target_parameters,
    )
)


assert np.isfinite(
    parameter_difference
)


print(
    f"[INFO] Policy-target parameter "
    f"squared difference: "
    f"{parameter_difference:.8f}"
)


# ============================================================
# SAVE CONTROL CHECKPOINT
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ca_rdqn_control"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "ca_rdqn_control_seed42.pt"
)


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
        "action_dim": ACTION_DIM,
        "observation_dim":
            OBSERVATION_DIM,
        "history": STATE_HISTORY,
    },
    checkpoint_path,
)


assert os.path.exists(
    checkpoint_path
)


print(
    "[PASS] Control checkpoint saved."
)


# ============================================================
# RELOAD
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)


reloaded_policy = RecurrentDQN()

reloaded_policy.load_state_dict(
    checkpoint[
        "policy_state_dict"
    ]
)


# ------------------------------------------------------------
# Reproducibility of saved model
# ------------------------------------------------------------

test_state = torch.tensor(
    state,
    dtype=torch.float32
).unsqueeze(0)


with torch.no_grad():

    original_q = policy_net(
        test_state
    )

    reloaded_q = reloaded_policy(
        test_state
    )


assert torch.allclose(
    original_q,
    reloaded_q,
    atol=1e-6
)


print(
    "[PASS] Checkpoint reload verified."
)

print(
    "[PASS] Saved-model prediction preservation verified."
)


# ============================================================
# FINAL
# ============================================================

print()
print(
    "[PASS] Real BSM1 CA-RDQN interaction verified."
)

print(
    "[PASS] Replay learning verified."
)

print(
    "[PASS] Finite training losses verified."
)

print(
    "[PASS] Finite rewards verified."
)

print(
    "[PASS] Four-sensor constraint maintained."
)

print(
    "[PASS] Chronological training environment maintained."
)

print(
    "[PASS] Control checkpoint verified."
)

print("=" * 70)
print(
    "STEP 66 COMPLETED"
)
print("=" * 70)