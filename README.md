# BrainBlock DRL — CS 445 Spring 2026

Tetromino packing puzzle solved with **Soft Actor-Critic (SAC)** using PyTorch.

## Project Structure

```text
brainblock/
├── environment.py      – Gymnasium-compatible 8×5 BrainBlock environment
├── sac_agent.py        – Discrete SAC with action masking + auto-entropy
├── train.py            – Training script (configurable hyperparameters)
├── evaluate.py         – Deterministic rollouts, metrics, solution plots, GIF export
├── plot_results.py     – Learning-curve figures from CSV logs
├── visualize_gif.py    – Visualize generated solution GIFs
├── demo_ui.py          – Interactive UI for watching trained agents
├── demo_run.py         – Smoke tests and quick demonstrations
├── requirements.txt    – Project dependencies
└── README.md
```

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements:**

* Python ≥ 3.10
* CUDA (optional, but recommended for faster training)

---

## Quick Smoke Test

Verify that the environment and agent are working correctly:

```bash
python demo_run.py
```

---

## Training

### Dense Reward (Recommended)

```bash
python train.py --reward_fn dense --seed 0 --episodes 30000
```

### Sparse Reward (Comparison)

```bash
python train.py --reward_fn sparse --seed 0 --episodes 30000
```

Training outputs are stored under:

```text
runs/
└── sac_<reward>_seed<n>/
    ├── log.csv
    └── checkpoints/
        └── final.pt
```

---

## Key Hyperparameters

| Flag             | Default   | Description                                      |
| ---------------- | --------- | ------------------------------------------------ |
| `--reward_fn`    | `dense`   | Reward function (`dense` or `sparse`)            |
| `--episodes`     | `20000`   | Number of training episodes                      |
| `--hidden`       | `256 256` | Hidden layer sizes for actor and critic networks |
| `--lr`           | `3e-4`    | Learning rate                                    |
| `--gamma`        | `0.99`    | Discount factor                                  |
| `--tau`          | `5e-3`    | Polyak averaging coefficient                     |
| `--batch_size`   | `256`     | Replay buffer batch size                         |
| `--buffer_cap`   | `100000`  | Replay buffer capacity                           |
| `--warmup_steps` | `2000`    | Environment steps before learning begins         |

---

## Evaluation

Evaluate a trained checkpoint and optionally generate GIF visualizations:

```bash
python evaluate.py --ckpt runs/sac_dense_seed0/checkpoints/final.pt --reward_fn dense --n_episodes 500 --n_seeds 5 --max_solutions 5 --out_dir eval_out/dense_seed0 --gif
```

### Evaluation Outputs

```text
eval_out/
└── dense_seed0/
    ├── eval_results.csv
    ├── solution_01.png
    ├── solution_02.png
    ├── ...
    └── gifs/
        ├── solution_01.gif
        ├── solution_02.gif
        └── ...
```

Generated files:

| File                   | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `eval_results.csv`     | Per-episode evaluation metrics and summary statistics    |
| `solution_XX.png`      | Color-coded final board layouts for successful solutions |
| `gifs/solution_XX.gif` | Animated visualization of the piece-placement process    |

---

## Interactive Demo UI

Launch the interactive visualization interface:

```bash
python demo_ui.py
```

### Default Settings

| Parameter      | Value                                       |
| -------------- | ------------------------------------------- |
| Checkpoint     | `runs/sac_dense_seed0/checkpoints/final.pt` |
| Playback Speed | `0.5`                                       |

### Custom Example

```bash
python demo_ui.py --ckpt runs\sac_dense_seed0\checkpoints\final.pt --speed 3.0
```

---

## Plotting Training Results

### Compare Dense and Sparse Training

```bash
python plot_results.py --log_dirs runs/sac_dense_seed0/log.csv runs/sac_sparse_seed0/log.csv --window 200 --out_dir plots/
```

Produces:

```text
plots/
├── training_curves.png
└── invalid_action_rate.png
```

### Plot All Dense-Reward Seeds

```bash
python plot_results.py --log_dirs runs/sac_dense_seed0/log.csv runs/sac_dense_seed1/log.csv runs/sac_dense_seed2/log.csv runs/sac_dense_seed3/log.csv runs/sac_dense_seed4/log.csv --out_dir plots/dense_all_seeds
```

Produces:

```text
plots/
└── dense_all_seeds/
    ├── training_curves.png
    └── invalid_action_rate.png
```

Files generated:

* `training_curves.png` — reward, covered cells, episode length, and success-rate curves.
* `invalid_action_rate.png` — invalid-action frequency during training.

---

## MDP Design

### State Observation (Dimension = 50)

| Component        | Size | Description                                |
| ---------------- | ---- | ------------------------------------------ |
| Board            | 40   | Binary occupancy grid (row-major order)    |
| Current Piece    | 5    | One-hot encoding of `{I, O, L, Z, T}`      |
| Remaining Counts | 5    | Normalized counts of remaining tetrominoes |

### Action Space

```text
A = {0..7} × {0..7} × {0..4}
```

Flattened into:

```text
[0, 319]
```

Representing:

```text
(orientation, x_anchor, y_anchor)
```

---

## Reward Functions

### Dense Reward (Recommended)

* `+0.4` per occupied cell placed successfully
* `+0.2` adjacency bonus when touching existing filled cells
* `+10` upon complete board coverage
* `−1` for invalid actions (episode terminates)

### Sparse Reward

* `+10` upon complete board coverage
* `−1` for invalid actions (episode terminates)
* `0` otherwise

---

## Why Soft Actor-Critic (SAC)?

Soft Actor-Critic is a maximum-entropy, off-policy reinforcement learning algorithm that:

1. Encourages exploration through entropy regularization.
2. Handles sparse rewards effectively using experience replay.
3. Automatically adapts the temperature parameter (`α`) to balance exploration and exploitation.

Additionally, action masking prevents the agent from selecting geometrically invalid placements, significantly improving sample efficiency and convergence speed.

---

## Academic Integrity

AI tools were used to assist with code organization and documentation drafting. All reinforcement learning formulation, environment design, reward engineering, implementation decisions, and experimental methodology are the authors' own work.
