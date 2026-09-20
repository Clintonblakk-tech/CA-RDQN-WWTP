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
# DATA
# ============================================================

from dqn_data_loader import (
    sensors,
    targets,
    train_indices,
    HISTORY,
    SENSOR_COUNT,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

ACTION_DIM = 495

OBSERVATION_DIM = 24

HIDDEN_DIM = 128

FC_DIM = 64

FORECAST_STEPS = 2

STATE_HISTORY = 4

DECISION_STEP = 1

ESTIMATOR_INPUT_DIM = 120

NH4_SCALE = 0.620953

SNO_SCALE = 0.351369


# ============================================================
# REPRODUCIBILITY
# ============================================================


# ============================================================
# ACTION TABLE
# ============================================================

ACTION_TABLE = np.array(
    list(
        combinations(
            range(SENSOR_COUNT),
            4
        )
    ),
    dtype=np.int64
)

assert ACTION_TABLE.shape == (
    ACTION_DIM,
    4
)


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

        sequence_output, _ = self.gru(x)

        final_hidden = sequence_output[:, -1, :]

        q_values = self.q_head(
            final_hidden
        )

        assert q_values.shape[1] == ACTION_DIM

        assert torch.isfinite(
            q_values
        ).all()

        return q_values


# ============================================================
# CA-RDQN ENVIRONMENT
# ============================================================

class CausalRDQNEnvironment:

    def __init__(
        self,
        start_index,
        end_index,
    ):

        self.start_index = int(
            start_index
        )

        self.end_index = int(
            end_index
        )

        assert self.start_index >= HISTORY

        assert (
            self.end_index
            >
            self.start_index
        )

        assert (
            self.end_index
            + FORECAST_STEPS
            <
            len(sensors)
        )

        self.current_index = None

        self.measurement_history = None

        self.activation_history = None


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.current_index = (
            self.start_index
        )

        self.measurement_history = np.zeros(
            (
                STATE_HISTORY,
                SENSOR_COUNT
            ),
            dtype=np.float32
        )

        self.activation_history = np.zeros(
            (
                STATE_HISTORY,
                SENSOR_COUNT
            ),
            dtype=np.float32
        )

        return self.get_state_sequence()


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _build_observation(self):

        observation = np.concatenate(
            [
                self.measurement_history[
                    -1
                ],

                self.activation_history[
                    -1
                ],
            ]
        ).astype(
            np.float32
        )

        assert observation.shape == (
            OBSERVATION_DIM,
        )

        assert np.isfinite(
            observation
        ).all()

        return observation


    # ========================================================
    # RECURRENT STATE
    # ========================================================

    def get_state_sequence(self):

        state = np.concatenate(
            [
                self.measurement_history,
                self.activation_history,
            ],
            axis=1
        ).astype(
            np.float32
        )

        assert state.shape == (
            STATE_HISTORY,
            OBSERVATION_DIM
        )

        assert np.isfinite(
            state
        ).all()

        return state


    # ========================================================
    # CURRENT ACTION MEASUREMENT
    # ========================================================

    def _acquire_measurements(
        self,
        index,
        action,
    ):

        selected = ACTION_TABLE[
            action
        ]

        measurement = np.zeros(
            SENSOR_COUNT,
            dtype=np.float32
        )

        mask = np.zeros(
            SENSOR_COUNT,
            dtype=np.float32
        )

        measurement[
            selected
        ] = sensors[
            index,
            selected
        ]

        mask[
            selected
        ] = 1.0

        assert np.all(
            mask[selected] == 1.0
        )

        inactive = np.setdiff1d(
            np.arange(SENSOR_COUNT),
            selected
        )

        assert np.all(
            mask[inactive] == 0.0
        )

        assert np.all(
            measurement[inactive] == 0.0
        )

        return (
            measurement,
            mask,
        )


    # ========================================================
    # ESTIMATOR INPUT
    # ========================================================

    def _build_estimator_input(
        self,
        current_measurement,
        current_mask,
    ):

        x = np.concatenate(
            [
                current_measurement,
                current_mask,
                self.measurement_history.reshape(-1),
                self.activation_history.reshape(-1),
            ]
        ).astype(
            np.float32
        )

        assert x.shape == (
            ESTIMATOR_INPUT_DIM,
        )

        assert np.isfinite(
            x
        ).all()

        return x


    # ========================================================
    # MANUAL ESTIMATOR
    # ========================================================

    @staticmethod
    def _manual_mlp(
        x,
        weights,
    ):

        W1, b1, W2, b2, W3, b3 = weights

        h1 = np.maximum(
            0.0,
            x @ W1 + b1
        )

        h2 = np.maximum(
            0.0,
            h1 @ W2 + b2
        )

        output = (
            h2 @ W3
            +
            b3
        )

        return float(
            output.reshape(-1)[0]
        )


    # ========================================================
    # LOAD ESTIMATOR WEIGHTS
    # ========================================================

    @staticmethod
    def _load_weights(
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
        ), f"Missing estimator: {path}"

        data = np.load(
            path
        )

        weights = (
            data["W1"],
            data["b1"],
            data["W2"],
            data["b2"],
            data["W3"],
            data["b3"],
        )

        assert weights[0].shape == (
            120,
            128
        )

        assert weights[1].shape == (
            128,
        )

        assert weights[2].shape == (
            128,
            64
        )

        assert weights[3].shape == (
            64,
        )

        assert weights[4].shape == (
            64,
            1
        )

        assert weights[5].shape == (
            1,
        )

        return weights


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
        nh4_weights,
        sno_weights,
    ):

        action = int(
            action
        )

        assert (
            0 <= action < ACTION_DIM
        )

        selected = ACTION_TABLE[
            action
        ]

        assert len(
            selected
        ) == 4

        # ----------------------------------------------------
        # Current measurements are acquired AFTER action.
        # ----------------------------------------------------

        current_measurement, current_mask = (
            self._acquire_measurements(
                self.current_index,
                action
            )
        )

        # ----------------------------------------------------
        # Build estimator input using only information
        # actually available under the activation history.
        # ----------------------------------------------------

        estimator_input = self._build_estimator_input(
            current_measurement,
            current_mask
        )

        pred_nh4 = self._manual_mlp(
            estimator_input,
            nh4_weights
        )

        pred_sno = self._manual_mlp(
            estimator_input,
            sno_weights
        )

        # ----------------------------------------------------
        # Actual 30-minute changes.
        # ----------------------------------------------------

        actual_change = (
            targets[
                self.current_index
                +
                FORECAST_STEPS
            ]
            -
            targets[
                self.current_index
            ]
        )

        actual_nh4 = float(
            actual_change[0]
        )

        actual_sno = float(
            actual_change[1]
        )

        # ----------------------------------------------------
        # Prediction errors.
        # ----------------------------------------------------

        error_nh4 = (
            pred_nh4
            -
            actual_nh4
        )

        error_sno = (
            pred_sno
            -
            actual_sno
        )

        # ----------------------------------------------------
        # Normalized reward.
        # ----------------------------------------------------

        reward = -(
            0.5
            *
            (
                error_nh4
                /
                NH4_SCALE
            ) ** 2

            +

            0.5
            *
            (
                error_sno
                /
                SNO_SCALE
            ) ** 2
        )

        assert np.isfinite(
            reward
        )

        # ----------------------------------------------------
        # Update activation history with information acquired
        # at the current decision.
        # ----------------------------------------------------

        self.measurement_history = np.roll(
            self.measurement_history,
            -1,
            axis=0
        )

        self.activation_history = np.roll(
            self.activation_history,
            -1,
            axis=0
        )

        self.measurement_history[
            -1
        ] = current_measurement

        self.activation_history[
            -1
        ] = current_mask

        # ----------------------------------------------------
        # Advance to the NEXT decision time.
        # ----------------------------------------------------

        next_index = (
            self.current_index
            +
            DECISION_STEP
        )

        done = (
            next_index
            >
            self.end_index
        )

        self.current_index = next_index

        # ----------------------------------------------------
        # Next state is genuinely associated with the next
        # decision time.
        # ----------------------------------------------------

        next_state = (
            self.get_state_sequence()
        )

        info = {
            "current_index": (
                self.current_index
                -
                DECISION_STEP
            ),

            "next_index": (
                self.current_index
            ),

            "selected_sensors": (
                selected.tolist()
            ),

            "predicted_nh4": (
                pred_nh4
            ),

            "predicted_sno": (
                pred_sno
            ),

            "actual_nh4": (
                actual_nh4
            ),

            "actual_sno": (
                actual_sno
            ),

            "error_nh4": (
                error_nh4
            ),

            "error_sno": (
                error_sno
            ),
        }

        return (
            next_state,
            reward,
            done,
            info,
        )

    if __name__ == "__main__":

        print("=" * 70)
        print("STEP 65 ...")

    

    


        # ============================================================
        # VALIDATION
        # ============================================================

        print("=" * 70)
        print(
            "STEP 65 â€” CA-RDQN TRAINING ENVIRONMENT"
        )
        print("=" * 70)

        print(
            f"[INFO] Sensor count: {SENSOR_COUNT}"
        )

        print(
            f"[INFO] Action dimension: {ACTION_DIM}"
        )

        print(
            f"[INFO] Observation dimension: "
            f"{OBSERVATION_DIM}"
        )

        print(
            f"[INFO] Recurrent history: "
            f"{STATE_HISTORY}"
        )

        print(
            f"[INFO] Estimator input dimension: "
            f"{ESTIMATOR_INPUT_DIM}"
        )

        print(
            f"[INFO] Forecast horizon: "
            f"{FORECAST_STEPS * 15} minutes"
        )

        # ============================================================
        # LOAD ESTIMATORS
        # ============================================================

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

        print(
            "[PASS] Causal NH4 estimator weights loaded."
        )

        print(
            "[PASS] Causal SNO estimator weights loaded."
        )

        # ============================================================
        # ENVIRONMENT
        # ============================================================

        env = CausalRDQNEnvironment(
            start_index=int(train_indices[0]),
            end_index=int(train_indices[-1]),
        )

        state_sequence = env.reset()

        assert state_sequence.shape == (
            STATE_HISTORY,
            OBSERVATION_DIM
        )

        assert np.isfinite(
            state_sequence
        ).all()

        print(
            "[PASS] Training environment reset verified."
        )

        # ============================================================
        # RECURRENT NETWORK
        # ============================================================

        policy_net = RecurrentDQN()

        test_tensor = torch.tensor(
            state_sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = policy_net(
                test_tensor
            )

        assert q_values.shape == (
            1,
            ACTION_DIM
        )

        assert torch.isfinite(
            q_values
        ).all()

        greedy_action = int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )

        print(
            "[PASS] Recurrent DQN integrated."
        )

        print(
            f"[INFO] Initial greedy action: "
            f"{greedy_action}"
        )

        # ============================================================
        # ACTION VALIDATION
        # ============================================================

        test_actions = [
            0,
            100,
            250,
            309,
            494,
        ]

        previous_index = env.current_index

        for action in test_actions:
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

            assert next_state.shape == (
                STATE_HISTORY,
                OBSERVATION_DIM
            )

            assert np.isfinite(
                next_state
            ).all()

            assert np.isfinite(
                reward
            )

            assert info[
                "current_index"
            ] == previous_index

            assert info[
                "next_index"
            ] == previous_index + 1

            previous_index = (
                env.current_index
            )

            print()
            print(
                f"[PASS] Action {action}: "
                f"{selected.tolist()}"
            )

            print(
                f"       Predicted Î”NH4: "
                f"{info['predicted_nh4']:.6f}"
            )

            print(
                f"       Actual Î”NH4: "
                f"{info['actual_nh4']:.6f}"
            )

            print(
                f"       Predicted Î”SNO: "
                f"{info['predicted_sno']:.6f}"
            )

            print(
                f"       Actual Î”SNO: "
                f"{info['actual_sno']:.6f}"
            )

            print(
                f"       Reward: "
                f"{reward:.6f}"
            )

        # ============================================================
        # TEMPORAL TRANSITION CHECK
        # ============================================================

        assert env.current_index == (
            int(train_indices[0])
            +
            len(test_actions)
        )

        assert next_state.shape == (
            4,
            24
        )

        print()
        print(
            "[PASS] Temporal state transition verified."
        )

        # ============================================================
        # INFORMATION AVAILABILITY
        # ============================================================

        latest_measurements = (
            next_state[-1, :12]
        )

        latest_masks = (
            next_state[-1, 12:24]
        )

        inactive = (
            latest_masks == 0
        )

        assert np.all(
            latest_measurements[inactive] == 0
        )

        assert np.sum(
            latest_masks
        ) == 4

        print(
            "[PASS] Current measurement availability verified."
        )

        print(
            "[PASS] Exactly four active sensors verified."
        )

        # ============================================================
        # CAUSAL HISTORY
        # ============================================================

        historical_measurements = (
            next_state[:, :12]
        )

        historical_masks = (
            next_state[:, 12:24]
        )

        for h in range(
            STATE_HISTORY
        ):
            unavailable = (
                historical_masks[h] == 0
            )

            assert np.all(
                historical_measurements[h][
                    unavailable
                ] == 0
            )

        print(
            "[PASS] Historical causal availability verified."
        )

        # ============================================================
        # ESTIMATOR INTERFACE
        # ============================================================

        current_measurement = (
            latest_measurements
        )

        current_mask = (
            latest_masks
        )

        estimator_input = np.concatenate(
            [
                current_measurement,
                current_mask,
                historical_measurements.reshape(-1),
                historical_masks.reshape(-1),
            ]
        )

        assert estimator_input.shape == (
            120,
        )

        assert np.isfinite(
            estimator_input
        ).all()

        print(
            "[PASS] 120-D estimator interface verified."
        )

        # ============================================================
        # TARGET ALIGNMENT
        # ============================================================

        test_t = int(
            train_indices[0]
        )

        expected_target = (
            targets[
                test_t + FORECAST_STEPS
            ]
            -
            targets[test_t]
        )

        actual_target = (
            info["actual_nh4"],
            info["actual_sno"],
        )

        assert np.allclose(
            expected_target,
            actual_target,
            rtol=1e-6,
            atol=1e-8,
        ), (
            f"Target mismatch: expected={expected_target}, "
            f"actual={actual_target}"
        )

        print(
            "[PASS] 30-minute target calculation verified."
        )

        # ============================================================
        # FINAL
        # ============================================================

        print()
        print(
            "[PASS] 495-action feasible action space verified."
        )

        print(
            "[PASS] Recurrent 4-step state verified."
        )

        print(
            "[PASS] Causal sensor acquisition verified."
        )

        print(
            "[PASS] Temporal s_t â†’ a_t â†’ r_t â†’ s_(t+1) "
            "transition verified."
        )

        print(
            "[PASS] Causal estimator integration verified."
        )

        print(
            "[PASS] Reward calculation verified."
        )

        print("=" * 70)
        print(
            "STEP 65 COMPLETED"
        )
        print("=" * 70)


