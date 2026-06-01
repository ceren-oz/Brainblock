"""
Training script for BrainBlock SAC agent.

Usage:
    python train.py --reward_fn dense --seed 0 --episodes 20000

Produces:
    checkpoints/   – model snapshots
    logs/          – per-episode CSV logs
"""

import argparse
import os
import csv
import time
import random
import numpy as np
import torch

from environment import BrainBlockEnv
from sac_agent import SACAgent

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SAC on BrainBlock")
    p.add_argument("--reward_fn",     type=str,   default="dense",
                   choices=["sparse", "dense"], help="Reward function variant")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--episodes",      type=int,   default=20_000)
    p.add_argument("--hidden",        type=int,   nargs="+", default=[256, 256])
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--tau",           type=float, default=5e-3)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--buffer_cap",    type=int,   default=100_000)
    p.add_argument("--warmup_steps",  type=int,   default=2_000)
    p.add_argument("--update_every",  type=int,   default=1)
    p.add_argument("--save_every",    type=int,   default=2_000)
    p.add_argument("--log_every",     type=int,   default=200)
    p.add_argument("--out_dir",       type=str,   default="runs")
    return p.parse_args()


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)

    run_name = f"sac_{args.reward_fn}_seed{args.seed}"
    out_dir  = os.path.join(args.out_dir, run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Environment
    env = BrainBlockEnv(reward_fn=args.reward_fn)
    obs_dim  = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    # Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        obs_dim      = obs_dim,
        n_actions    = n_actions,
        hidden       = tuple(args.hidden),
        lr           = args.lr,
        gamma        = args.gamma,
        tau          = args.tau,
        buffer_capacity = args.buffer_cap,
        batch_size   = args.batch_size,
        warmup_steps = args.warmup_steps,
        update_every = args.update_every,
        device       = device,
    )

    print(f"[{run_name}] device={device}  obs_dim={obs_dim}  n_actions={n_actions}")

    # CSV logger
    csv_path = os.path.join(out_dir, "log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow([
        "episode", "reward", "ep_len", "covered", "solved",
        "invalid_actions", "invalid_rate",
        "critic_loss", "actor_loss", "alpha",
    ])

    # Metrics accumulators
    ep_rewards, ep_lengths, ep_solved = [], [], []
    ep_covered, ep_invalid_rate = [], []
    recent_rewards = []
    t0 = time.time()

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        mask   = env.get_valid_action_mask()
        total_reward = 0.0
        last_info    = {}
        update_info  = {}

        while True:
            action    = agent.select_action(obs, mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = env.get_valid_action_mask()
            done      = terminated or truncated

            agent.store(obs, action, reward, next_obs, done, mask, next_mask)
            ui = agent.update()
            if ui:
                update_info = ui

            total_reward += reward
            obs, mask = next_obs, next_mask
            last_info = info
            if done:
                break

        n_steps  = env.step_count + env.invalid_action_count
        inv_rate = env.invalid_action_count / max(n_steps, 1)

        ep_rewards.append(total_reward)
        ep_lengths.append(n_steps)
        ep_solved.append(int(last_info.get("solved", False)))
        ep_covered.append(last_info.get("covered", 0))
        ep_invalid_rate.append(inv_rate)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 200:
            recent_rewards.pop(0)

        writer.writerow([
            episode, total_reward, n_steps,
            last_info.get("covered", 0),
            int(last_info.get("solved", False)),
            env.invalid_action_count, f"{inv_rate:.3f}",
            update_info.get("critic_loss", ""),
            update_info.get("actor_loss", ""),
            update_info.get("alpha", ""),
        ])
        csv_file.flush()

        if episode % args.log_every == 0:
            sr     = np.mean(ep_solved[-args.log_every:])
            mean_r = np.mean(ep_rewards[-args.log_every:])
            elapsed = time.time() - t0
            print(
                f"Ep {episode:6d} | "
                f"R={mean_r:+.2f} | "
                f"SR={sr:.2%} | "
                f"cov={np.mean(ep_covered[-args.log_every:]):.1f}/40 | "
                f"inv={np.mean(ep_invalid_rate[-args.log_every:]):.2%} | "
                f"α={update_info.get('alpha', 0):.4f} | "
                f"t={elapsed:.0f}s"
            )

        if episode % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ep{episode}.pt")
            agent.save(ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path}")

    # Final save
    agent.save(os.path.join(ckpt_dir, "final.pt"))
    csv_file.close()
    print(f"\nTraining complete. Logs: {csv_path}")
    return out_dir, agent


if __name__ == "__main__":
    args = parse_args()
    train(args)
