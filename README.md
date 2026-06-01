# BrainBlock DRL — CS 445 Spring 2026

Tetromino packing puzzle solved with **Soft Actor-Critic (SAC)** using PyTorch.

## Project structure

```
brainblock/
├── environment.py   – Gymnasium-compatible 8×5 BrainBlock env
├── sac_agent.py     – Discrete SAC with action masking + auto-entropy
├── train.py         – Training script (configurable hyperparameters)
├── evaluate.py      – Deterministic rollouts, metrics, solution plots
├── plot_results.py  – Learning-curve figures from CSV logs
├── demo_run.py      – Smoke tests + 500-ep quick demo
└── README.md
```

## Installation

```bash
pip install torch gymnasium numpy matplotlib
```

Python ≥ 3.10 recommended.  CUDA is optional but speeds up training.

## Quick smoke test

```bash
python demo_run.py
```

## Training

### Dense reward (recommended)

```bash
python train.py --reward_fn dense --seed 0 --episodes 30000
```

### Sparse reward (comparison)

```bash
python train.py --reward_fn sparse --seed 0 --episodes 30000
```

### Multi-seed run (5 seeds)

```bash
for seed in 0 1 2 3 4; do
    python train.py --reward_fn dense --seed $seed --episodes 30000
done
```

Checkpoints are saved to `runs/sac_<reward>_seed<n>/checkpoints/`.  
Logs (CSV) are at `runs/sac_<reward>_seed<n>/log.csv`.

### Key hyperparameters

| Flag            | Default    | Description                          |
|-----------------|------------|--------------------------------------|
| `--reward_fn`   | `dense`    | `dense` or `sparse`                  |
| `--episodes`    | `20000`    | Training episodes                    |
| `--hidden`      | `256 256`  | MLP hidden layer sizes               |
| `--lr`          | `3e-4`     | Learning rate (actor + critic)       |
| `--gamma`       | `0.99`     | Discount factor                      |
| `--tau`         | `5e-3`     | Polyak averaging rate                |
| `--batch_size`  | `256`      | Replay buffer batch size             |
| `--buffer_cap`  | `100000`   | Replay buffer capacity               |
| `--warmup_steps`| `2000`     | Steps before first gradient update   |

## Evaluation

```bash
python evaluate.py \
    --ckpt runs/sac_dense_seed0/checkpoints/final.pt \
    --reward_fn dense \
    --n_episodes 500 \
    --n_seeds 5 \
    --max_solutions 10 \
    --out_dir eval_out/
```

Outputs:
- `eval_out/eval_results.csv` — per-episode metrics
- `eval_out/solution_01.png` … `solution_10.png` — colour-coded board renders

## Plotting

```bash
python plot_results.py \
    --log_dirs runs/sac_dense_seed0/log.csv \
              runs/sac_sparse_seed0/log.csv \
    --window 200 \
    --out_dir plots/
```

Produces:
- `plots/training_curves.png`  — reward, covered cells, episode length, success rate
- `plots/invalid_action_rate.png`

---

## MDP Design

### State observation (dim = 50)

| Component | Size | Description |
|-----------|------|-------------|
| Board     | 40   | Binary grid (row-major), 1 = filled |
| Current piece | 5 | One-hot over {I, O, L, Z, T} |
| Remaining counts | 5 | Normalised count of each piece still in queue |

### Action space

`A = {0..7} × {0..7} × {0..4}` → flat index `[0, 319]`  
`(orientation, x_anchor, y_anchor)`

### Reward functions

**Dense (recommended):**
- `+0.4` per cell placed (up to `+1.6` per step)
- `+0.2` adjacency bonus when piece is placed touching an existing filled cell
- `+10` on full-board completion
- `−1` on invalid action (hard termination)

**Sparse:**
- `+10` on full-board completion
- `−1` on invalid action (hard termination)
- `0` otherwise

### Why SAC?

SAC is a maximum-entropy off-policy method that naturally:
1. **Explores** via entropy regularisation — avoids deterministic collapse early in training.
2. **Handles sparse rewards** better than on-policy methods due to experience replay.
3. **Adapts** the temperature `α` automatically, balancing exploration vs exploitation.

Action masking prevents the agent from wasting gradient steps on geometrically impossible placements, dramatically accelerating convergence.

---

## Academic Integrity

AI tools used for code structure and documentation drafting; all RL formulation, reward design, and experimental choices are the authors' own work.
