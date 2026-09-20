import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT NETWORK
# ============================================================

from recurrent_dqn_network import (
    RecurrentDQN,
    OBSERVATION_DIM,
    HISTORY,
    ACTION_DIM,
)


# ============================================================
# CONFIGURATION
# ============================================================

STATE_DIM = OBSERVATION_DIM * HISTORY

GAMMA = 0.99

LEARNING_RATE = 1e-3

BATCH_SIZE = 64

REPLAY_CAPACITY = 50_000

TARGET_UPDATE_FREQUENCY = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 20_000

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# REPLAY BUFFER
# ============================================================

class RecurrentReplayBuffer:

    def __init__(
        self,
        capacity,
    ):

        self.capacity = int(
            capacity
        )

        self.buffer = []

        self.position = 0


    def push(
        self,
        state_sequence,
        action,
        reward,
        next_state_sequence,
        done,
    ):

        state_sequence = np.asarray(
            state_sequence,
            dtype=np.float32
        )

        next_state_sequence = np.asarray(
            next_state_sequence,
            dtype=np.float32
        )

        assert state_sequence.shape == (
            HISTORY,
            OBSERVATION_DIM
        )

        assert next_state_sequence.shape == (
            HISTORY,
            OBSERVATION_DIM
        )

        transition = (
            state_sequence,
            int(action),
            float(reward),
            next_state_sequence,
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

        assert (
            len(self.buffer)
            >= batch_size
        )

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
            dones,
        )


    def __len__(self):

        return len(
            self.buffer
        )


# ============================================================
# EPSILON SCHEDULE
# ============================================================

def epsilon_by_step(step):

    step = int(step)

    fraction = min(
        step / EPSILON_DECAY_STEPS,
        1.0
    )

    epsilon = (
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

    return float(epsilon)


# ============================================================
# EPSILON-GREEDY
# ============================================================

def select_action(
    network,
    state_sequence,
    epsilon,
):

    if random.random() < epsilon:

        return random.randrange(
            ACTION_DIM
        )

    state_tensor = torch.tensor(
        state_sequence,
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
# DOUBLE-DQN TARGET
# ============================================================

def double_dqn_target(
    policy_net,
    target_net,
    next_states,
    rewards,
    dones,
):

    with torch.no_grad():

        # ----------------------------------------------------
        # Online network selects the next action.
        # ----------------------------------------------------

        next_q_policy = policy_net(
            next_states
        )

        next_actions = torch.argmax(
            next_q_policy,
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Target network evaluates that action.
        # ----------------------------------------------------

        next_q_target = target_net(
            next_states
        )

        selected_next_q = (
            next_q_target
            .gather(
                1,
                next_actions
            )
            .squeeze(1)
        )

        targets = (
            rewards
            +
            GAMMA
            *
            (
                1.0 - dones
            )
            *
            selected_next_q
        )

    return targets


# ============================================================
# TRAINING UPDATE
# ============================================================

def training_update(
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

    q_values = policy_net(
        states
    )

    selected_q = (
        q_values
        .gather(
            1,
            actions.unsqueeze(1)
        )
        .squeeze(1)
    )

    targets = double_dqn_target(
        policy_net,
        target_net,
        next_states,
        rewards,
        dones,
    )

    loss_function = nn.SmoothL1Loss()

    loss = loss_function(
        selected_q,
        targets
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
# VALIDATION
# ============================================================

print("=" * 70)
print(
    "STEP 62 — RECURRENT DQN TRAINING COMPONENTS"
)
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
    f"[INFO] Action dimension: "
    f"{ACTION_DIM}"
)

print(
    f"[INFO] Replay capacity: "
    f"{REPLAY_CAPACITY}"
)

print(
    f"[INFO] Batch size: "
    f"{BATCH_SIZE}"
)

print(
    f"[INFO] Gamma: "
    f"{GAMMA}"
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

print(
    "[PASS] Policy network initialized."
)

print(
    "[PASS] Target network initialized."
)

print(
    "[PASS] Target network synchronized."
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE
)

print(
    "[PASS] Adam optimizer initialized."
)


# ============================================================
# REPLAY BUFFER
# ============================================================

buffer = RecurrentReplayBuffer(
    REPLAY_CAPACITY
)


# ------------------------------------------------------------
# Insert synthetic transitions
# ------------------------------------------------------------

for i in range(
    BATCH_SIZE
):

    state_sequence = np.random.randn(
        HISTORY,
        OBSERVATION_DIM
    ).astype(
        np.float32
    )

    next_state_sequence = np.random.randn(
        HISTORY,
        OBSERVATION_DIM
    ).astype(
        np.float32
    )

    action = i % ACTION_DIM

    reward = -(
        float(i)
        /
        BATCH_SIZE
    )

    done = (
        i == BATCH_SIZE - 1
    )

    buffer.push(
        state_sequence,
        action,
        reward,
        next_state_sequence,
        done,
    )


assert len(buffer) == BATCH_SIZE

print(
    "[PASS] Replay buffer insertion verified."
)


# ============================================================
# SAMPLE
# ============================================================

(
    states,
    actions,
    rewards,
    next_states,
    dones,
) = buffer.sample(
    BATCH_SIZE
)

assert states.shape == (
    BATCH_SIZE,
    HISTORY,
    OBSERVATION_DIM
)

assert actions.shape == (
    BATCH_SIZE,
)

assert rewards.shape == (
    BATCH_SIZE,
)

assert next_states.shape == (
    BATCH_SIZE,
    HISTORY,
    OBSERVATION_DIM
)

assert dones.shape == (
    BATCH_SIZE,
)

print(
    "[PASS] Recurrent replay sampling verified."
)


# ============================================================
# EPSILON TEST
# ============================================================

epsilon_0 = epsilon_by_step(
    0
)

epsilon_5000 = epsilon_by_step(
    5000
)

epsilon_10000 = epsilon_by_step(
    10000
)

epsilon_20000 = epsilon_by_step(
    20000
)

assert np.isclose(
    epsilon_0,
    1.0
)

assert np.isclose(
    epsilon_5000,
    0.7625
)

assert np.isclose(
    epsilon_10000,
    0.525
)

assert np.isclose(
    epsilon_20000,
    0.05
)

assert (
    epsilon_0
    >
    epsilon_5000
    >
    epsilon_10000
    >
    epsilon_20000
)

print(
    "[PASS] Epsilon-greedy schedule verified."
)


# ============================================================
# ACTION TEST
# ============================================================

test_sequence = np.random.randn(
    HISTORY,
    OBSERVATION_DIM
).astype(
    np.float32
)

action_random = select_action(
    policy_net,
    test_sequence,
    epsilon=1.0
)

action_greedy = select_action(
    policy_net,
    test_sequence,
    epsilon=0.0
)

assert (
    0 <= action_random < ACTION_DIM
)

assert (
    0 <= action_greedy < ACTION_DIM
)

print(
    "[PASS] Epsilon-greedy action selection verified."
)


# ============================================================
# DOUBLE-DQN TARGET
# ============================================================

states_tensor = torch.tensor(
    states,
    dtype=torch.float32
)

rewards_tensor = torch.tensor(
    rewards,
    dtype=torch.float32
)

next_states_tensor = torch.tensor(
    next_states,
    dtype=torch.float32
)

dones_tensor = torch.tensor(
    dones,
    dtype=torch.float32
)

targets = double_dqn_target(
    policy_net,
    target_net,
    next_states_tensor,
    rewards_tensor,
    dones_tensor,
)

assert targets.shape == (
    BATCH_SIZE,
)

assert torch.isfinite(
    targets
).all()

print(
    "[PASS] Double-DQN Bellman targets verified."
)


# ============================================================
# TRAINING UPDATE
# ============================================================

loss = training_update(
    policy_net,
    target_net,
    optimizer,
    buffer,
)

assert np.isfinite(
    loss
)

print(
    f"[PASS] Training update verified "
    f"(loss = {loss:.6f})."
)


# ============================================================
# TARGET NETWORK UPDATE
# ============================================================

target_net.load_state_dict(
    policy_net.state_dict()
)

for policy_parameter, target_parameter in zip(
    policy_net.parameters(),
    target_net.parameters(),
):

    assert torch.allclose(
        policy_parameter,
        target_parameter
    )


print(
    "[PASS] Target-network synchronization verified."
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model_a = RecurrentDQN()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model_b = RecurrentDQN()

for parameter_a, parameter_b in zip(
    model_a.parameters(),
    model_b.parameters(),
):

    assert torch.allclose(
        parameter_a,
        parameter_b
    )


print(
    "[PASS] Training-network reproducibility verified."
)


# ============================================================
# FINAL
# ============================================================

print()
print(
    "[PASS] Recurrent replay buffer verified."
)

print(
    "[PASS] Double-DQN target verified."
)

print(
    "[PASS] Bellman learning update verified."
)

print(
    "[PASS] Target-network mechanism verified."
)

print(
    "[PASS] Epsilon schedule verified."
)

print(
    "[PASS] Reproducibility verified."
)

print("=" * 70)
print("STEP 62 COMPLETED")
print("=" * 70)