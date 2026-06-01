"""
Evaluation script for trained BrainBlock SAC agents.

Usage:
    python evaluate.py --ckpt runs/sac_dense_seed0/checkpoints/final.pt \
                       --reward_fn dense --n_episodes 500 --n_seeds 5

    # Also generate animated GIFs:
    python evaluate.py --ckpt runs/sac_dense_seed0/checkpoints/final.pt \
                       --reward_fn dense --n_episodes 500 --n_seeds 5 --gif

Outputs:
    - summary metrics (success rate, mean return, episode length, invalid-action rate)
    - up to N solution board renders (ASCII + matplotlib figure)
    - eval_results.csv
    - (optional) animated GIFs in <out_dir>/gifs/ when --gif is passed
"""

import argparse
import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from environment import BrainBlockEnv, PIECE_TYPES, INVENTORY, W, H
from sac_agent import SACAgent


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         type=str,   required=True)
    p.add_argument("--reward_fn",    type=str,   default="dense",
                   choices=["sparse", "dense"])
    p.add_argument("--n_episodes",   type=int,   default=500)
    p.add_argument("--n_seeds",      type=int,   default=5,
                   help="Random seeds for multi-seed eval")
    p.add_argument("--max_solutions",type=int,   default=10,
                   help="How many unique solutions to visualize")
    p.add_argument("--out_dir",      type=str,   default="eval_out")
    p.add_argument("--gif",          action="store_true",
                   help="Also render animated GIFs for each solution")
    p.add_argument("--gif_frame_ms", type=int,   default=420,
                   help="Frame duration in ms for GIFs (default 420)")
    p.add_argument("--gif_hold_ms",  type=int,   default=1800,
                   help="Final-frame hold duration in ms (default 1800)")
    return p.parse_args()


# -------------------------------------------------------------------------
# Board visualization helpers
# -------------------------------------------------------------------------

# Map piece types to colours for rendering
PIECE_COLORS = {
    "I": "#4C9BE8",
    "O": "#E8C44C",
    "L": "#4CE87A",
    "Z": "#E84C4C",
    "T": "#C44CE8",
    "_": "#EEEEEE",   # empty
}

def render_solution_matplotlib(board_history: list, piece_sequence: list,
                                title: str, save_path: str):
    """
    board_history: list of (H,W) arrays after each placement.
    piece_sequence: list of piece type strings in placement order.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Colour each cell by which piece placed it
    cell_color = [["#EEEEEE"] * W for _ in range(H)]
    prev = np.zeros((H, W), dtype=int)
    for step_idx, (board, piece) in enumerate(zip(board_history, piece_sequence)):
        diff = board - prev
        color = PIECE_COLORS.get(piece, "#888888")
        for r in range(H):
            for c in range(W):
                if diff[r, c] == 1:
                    cell_color[r][c] = color
        prev = board.copy()

    for r in range(H):
        for c in range(W):
            rect = mpatches.Rectangle(
                (c, r), 1, 1,
                linewidth=0.5, edgecolor="white",
                facecolor=cell_color[r][c]
            )
            ax.add_patch(rect)

    # Legend
    handles = [mpatches.Patch(color=PIECE_COLORS[p], label=p) for p in PIECE_TYPES]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.14, 1.0))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def board_to_key(board: np.ndarray) -> str:
    return board.tobytes().hex()


# -------------------------------------------------------------------------
# Rollout with board tracking
# -------------------------------------------------------------------------

def rollout(env: BrainBlockEnv, agent: SACAgent, deterministic: bool = True):
    obs, _ = env.reset()
    mask   = env.get_valid_action_mask()
    total_reward = 0.0
    boards   = []
    pieces   = []
    actions_taken = []

    while True:
        action = agent.select_action(obs, mask, deterministic=deterministic)
        actions_taken.append(action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = env.get_valid_action_mask()
        total_reward += reward
        done = terminated or truncated

        if info.get("solved") or (done and not info.get("invalid")):
            boards.append(env.board.copy())
            # piece placed = last piece that was popped (before step)
            # we track by re-reading sequence

        obs, mask = next_obs, next_mask
        if done:
            break

    return {
        "reward":     total_reward,
        "ep_len":     env.step_count + env.invalid_action_count,
        "covered":    info.get("covered", 0),
        "solved":     info.get("solved", False),
        "invalid":    env.invalid_action_count,
        "board":      env.board.copy(),
    }


# -------------------------------------------------------------------------
# Full episode with piece tracking (for visualisation)
# -------------------------------------------------------------------------

def rollout_with_history(env: BrainBlockEnv, agent: SACAgent, seed: int = 0):
    obs, _ = env.reset(seed=seed)
    original_queue = list(env.queue)   # capture before any pops
    mask   = env.get_valid_action_mask()
    board_history = []
    piece_sequence = []

    while True:
        piece = env.queue[0] if env.queue else None
        action = agent.select_action(obs, mask, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = env.get_valid_action_mask()

        if not info.get("invalid"):
            board_history.append(env.board.copy())
            piece_sequence.append(piece)

        obs, mask = next_obs, next_mask
        if terminated or truncated:
            break

    return info.get("solved", False), board_history, piece_sequence, env.board.copy(), original_queue


# -------------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------------

def evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = BrainBlockEnv(reward_fn=args.reward_fn)
    obs_dim  = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    agent = SACAgent(obs_dim=obs_dim, n_actions=n_actions, device=device)
    agent.load(args.ckpt)

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Evaluating {args.n_episodes} episodes × {args.n_seeds} seeds ...\n")

    all_rewards, all_lengths, all_solved = [], [], []
    all_covered, all_invalid_rates = [], []

    solutions = []   # unique solved boards

    csv_path = os.path.join(args.out_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "episode", "reward", "ep_len",
                         "covered", "solved", "invalid", "invalid_rate"])

        for seed_offset in range(args.n_seeds):
            base_seed = seed_offset * 10_000
            for ep in range(args.n_episodes):
                np.random.seed(base_seed + ep)
                res = rollout(env, agent, deterministic=False)

                ep_len   = res["ep_len"]
                inv_rate = res["invalid"] / max(ep_len, 1)

                all_rewards.append(res["reward"])
                all_lengths.append(ep_len)
                all_solved.append(int(res["solved"]))
                all_covered.append(res["covered"])
                all_invalid_rates.append(inv_rate)

                writer.writerow([
                    seed_offset, ep, res["reward"], ep_len,
                    res["covered"], int(res["solved"]),
                    res["invalid"], f"{inv_rate:.3f}",
                ])

                if res["solved"]:
                    key = board_to_key(res["board"])
                    if key not in [board_to_key(s[0]) for s in solutions]:
                        # Re-run to capture history
                        solved, hist, pseq, board, orig_queue = rollout_with_history(
                            env, agent, seed=base_seed + ep
                        )
                        if solved:
                            solutions.append((board, hist, pseq, orig_queue))

    # ---- Print summary ----
    print("=" * 55)
    print(f"Episodes evaluated : {len(all_rewards)}")
    print(f"Success rate       : {np.mean(all_solved):.2%}")
    print(f"Mean return (±std) : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean episode length: {np.mean(all_lengths):.1f}")
    print(f"Mean covered cells : {np.mean(all_covered):.1f} / 40")
    print(f"Invalid-action rate: {np.mean(all_invalid_rates):.2%}")
    print(f"Unique solutions   : {len(solutions)}")
    print("=" * 55)

    # ---- Visualise solutions ----
    for i, (board, hist, pseq, orig_queue) in enumerate(solutions[:args.max_solutions]):
        save_path = os.path.join(args.out_dir, f"solution_{i+1:02d}.png")
        render_solution_matplotlib(
            hist, pseq,
            title=f"BrainBlock Solution #{i+1}",
            save_path=save_path
        )
        print(f"Solution {i+1} saved → {save_path}")

        # Animated GIF
        if getattr(args, "gif", False):
            try:
                from visualize_gif import render_solution_gif
                gif_dir  = os.path.join(args.out_dir, "gifs")
                gif_path = os.path.join(gif_dir, f"solution_{i+1:02d}.gif")
                render_solution_gif(
                    hist, pseq, orig_queue,
                    title=f"Solution #{i+1}",
                    save_path=gif_path,
                    frame_duration_ms=getattr(args, "gif_frame_ms", 420),
                    final_hold_ms=getattr(args, "gif_hold_ms", 1800),
                )
            except Exception as exc:
                print(f"  [GIF skipped] {exc}")

        # ASCII fallback
        print(f"\n--- Solution {i+1} (ASCII) ---")
        for r in range(H):
            print("  " + " ".join("█" if board[r, c] else "·" for c in range(W)))
        print()

    print(f"\nCSV log: {csv_path}")
    return {
        "success_rate": np.mean(all_solved),
        "mean_return":  np.mean(all_rewards),
        "std_return":   np.std(all_rewards),
        "mean_length":  np.mean(all_lengths),
        "inv_rate":     np.mean(all_invalid_rates),
        "n_solutions":  len(solutions),
    }


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)