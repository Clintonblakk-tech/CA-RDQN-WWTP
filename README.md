# CA-RDQN-WWTP

## CA-RDQN-WWTP: Reproducibility Package for Dynamic Sensor Activation Scheduling in Wastewater Treatment Plants
This repository contains the code and reproducibility materials for the study:

>This reproducibility package provides the software, processed data, trained model checkpoints, experimental results and analysis scripts supporting the study of constraint-aware dynamic sensor activation scheduling in wastewater treatment plants. The package implements a constraint-aware recurrent deep Q-network (CA-RDQN) and a non-recurrent CA-DQN ablation for selecting exactly four sensors from twelve candidate process sensors at 15-min scheduling intervals using the BSM1 benchmark. It includes the frozen causal estimator used to predict 30-min changes in effluent ΔNH₄-N and SNO, the 495 feasible sensor configurations, multi-seed evaluation outputs, temporal sensor-selection analysis, statistical comparisons, and figure-generation scripts. The repository is intended to support reproducibility, inspection of the experimental workflow and reuse of the implementation. Original BSM1 benchmark input files are not redistributed and should be obtained from the authoritative BSM1 source.

## Study framework

- **Process benchmark:** BSM1
- **Process model:** ASM1
- **Candidate sensors:** 12
- **Active sensors per decision:** 4
- **Feasible sensor configurations:** 495
- **Decision interval:** 15 min
- **Prediction horizon:** 30 min
- **Historical state length:** 4 observations
- **Targets:** 30-min changes in effluent NH4-N and SNO
- **Principal agent:** constraint-aware recurrent DQN (CA-RDQN)
- **Ablation:** constraint-aware DQN (CA-DQN)
- **Final training seeds:** 11, 22, 33, 44, 55, 66, 77, 88, 99 and 111

## CA-RDQN

The CA-RDQN uses a four-step history of process measurements and sensor-availability indicators. A GRU with 128 hidden units encodes the temporal sequence, followed by a fully connected layer with 64 units and Q-values for all 495 feasible sensor configurations.

The final training protocol uses Double-DQN targets, a discount factor of 0.99, learning rate of 0.001, batch size of 64, replay capacity of 50,000, a warm-up period of 64 transitions, target-network updates every 500 optimisation steps, and an epsilon schedule from 1.0 to 0.05 over 20,000 steps.

## Causal estimator

A frozen causal estimator predicts the 30-min changes in NH4-N and SNO from measurements actually acquired by the scheduling policy. Historical measurements that were unavailable because a sensor was not activated are masked, while the corresponding availability indicators are retained.

The estimator input contains four historical 12-sensor measurement vectors and four corresponding availability masks, giving a 120-dimensional input. Separate estimators are used for the two prediction targets and are frozen before reinforcement-learning training.

## CA-DQN ablation

The CA-DQN ablation uses the same sensor pool, 495-action space, causal estimator, scheduling interval, prediction horizon, chronological split, training duration and ten-seed evaluation protocol. It replaces the recurrent four-step representation with the current 24-dimensional observation and fully connected layers of 128 and 64 units.

The gradient-clipping threshold is 1.0 for CA-DQN and 10.0 for CA-RDQN; this difference is retained explicitly in the repository configuration and should be considered when interpreting the ablation.

## Data and reproducibility

The repository contains processed BSM1 trajectories, estimator weights, trained CA-RDQN and CA-DQN checkpoints, seed-level results, statistical analyses, sensor-configuration benchmarks, temporal-oracle results and manuscript figure-generation materials.

The original BSM1 benchmark input files are not redistributed in this repository. Users should obtain the benchmark inputs from the authoritative BSM1 source and use the provided processing and export scripts where applicable.

The held-out evaluation period is kept separate from training. The final chronological split uses decision indices 4-862 for training and 863-1340 for evaluation.

## Repository structure

```text
CA-RDQN-WWTP/
├── config/
├── data/
│   └── processed/
├── outputs/
├── src/
├── .gitignore
├── README.md
├── requirements.txt
└── create_graphical_abstract.py
```

## Reproducibility

The final CA-RDQN and CA-DQN experiments use ten independent random seeds. Seed-level outputs are retained in `data/processed/final_ca_rdqn/` and `data/processed/final_ca_dqn_ablation/`.

The configuration in `config/config.yaml` documents the final experimental protocol. The executable implementation is provided in `src/`, including the data loader, causal environment, CA-RDQN training script, CA-DQN ablation script, validation scripts and analysis scripts.

## Licence and benchmark data

The repository licence and data-redistribution conditions should be checked before public release. The BSM1 benchmark input files are intentionally excluded from this repository.

## Citation

If you use this repository, please cite the associated research article and the archived repository release/DOI once available.

**Manuscript:** Constraint-Aware Deep Q-Network for Dynamic Sensor Activation Scheduling in Wastewater Treatment Plants: A Multi-Seed Robustness and Ablation Study

