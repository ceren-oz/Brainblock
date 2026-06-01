"""
Smoke test + quick demo training (500 episodes) to verify correctness.
Run: python demo_run.py
"""

import numpy as np
import torch
import os, sys

from environment import BrainBlockEnv, decode_action, PIECE_TRANSFORMS, W, H, N_ACTIONS
from sac_agent import SACAgent

# -----------------------------------------------------------------------
# 1. Environment smoke tests
# -----------------------------------------------------------------------
print("=" * 50)
print("1. Environment smoke tests")
print("=" * 50)

env = BrainBlockEnv(reward_fn="dense")
obs, info = env.reset(seed=42)
print(f"  obs shape     : {obs.shape}  (expected 50)")
print(f"  action space  : {env.action_space}  (expected Discrete(320))")
print(f"  queue length  : {len(env.queue)}  (expected 10)")
assert obs.shape == (50,), "Obs shape wrong"
assert env.action_space.n == 320, "Action space wrong"

# Valid-action mask
mask = env.get_valid_action_mask()
print(f"  valid actions : {mask.sum()} / {N_ACTIONS}")
assert mask.sum() > 0, "No valid actions on empty board — bug!"

# Step with a valid action
valid_idx = int(np.where(mask)[0][0])
obs2, rew, term, trunc, info2 = env.step(valid_idx)
print(f"  step result   : reward={rew:.2f}  covered={info2['covered']}")
assert info2["covered"] == 4, "Should cover 4 cells on first piece"

# Test invalid action is caught
obs, _ = env.reset(seed=1)
# Force an obviously invalid action (orientation 0, anchor 7,4 for I-piece likely OOB)
bad_action = decode_action(0)  # orient=0, x=0, y=0  -- may or may not be legal
# Try to overflow deliberately
bad_action = N_ACTIONS - 1
obs2, rew, term, trunc, info_bad = env.step(bad_action)
# Either legal or invalid — both fine; just shouldn't crash
print(f"  overflow action test passed (solved={info_bad.get('solved')}, invalid={info_bad.get('invalid')})")

print("  ✓ Environment smoke tests passed\n")

# -----------------------------------------------------------------------
# 2. Agent smoke test
# -----------------------------------------------------------------------
print("=" * 50)
print("2. SAC agent smoke tests")
print("=" * 50)

device = torch.device("cpu")
agent = SACAgent(obs_dim=50, n_actions=320, device=device,
                 warmup_steps=50, batch_size=32, buffer_capacity=500)

env2 = BrainBlockEnv(reward_fn="dense")
obs, _ = env2.reset(seed=0)
mask = env2.get_valid_action_mask()
action = agent.select_action(obs, mask)
print(f"  selected action : {action}  (valid={mask[action]})")

# Fill buffer past warmup and test update
for _ in range(100):
    a = agent.select_action(obs, mask)
    no, r, t, tr, inf = env2.step(a)
    nm = env2.get_valid_action_mask()
    agent.store(obs, a, r, no, t or tr, mask, nm)
    if t or tr:
        obs, _ = env2.reset()
        mask = env2.get_valid_action_mask()
    else:
        obs, mask = no, nm

ui = agent.update()
print(f"  update result   : {ui}")
print("  ✓ SAC agent smoke tests passed\n")

# -----------------------------------------------------------------------
# 3. Short training run (500 eps) to verify convergence direction
# -----------------------------------------------------------------------
print("=" * 50)
print("3. Short training demo (500 episodes, dense reward)")
print("=" * 50)

env3  = BrainBlockEnv(reward_fn="dense")
agent3 = SACAgent(obs_dim=50, n_actions=320, device=device,
                  warmup_steps=200, batch_size=64, buffer_capacity=10_000,
                  lr=3e-4, gamma=0.99)

rewards, covered_hist, solved_count = [], [], 0

for ep in range(1, 501):
    obs, _ = env3.reset(seed=ep)
    mask   = env3.get_valid_action_mask()
    tot_r  = 0.0
    while True:
        a = agent3.select_action(obs, mask)
        no, r, t, tr, inf = env3.step(a)
        nm = env3.get_valid_action_mask()
        agent3.store(obs, a, r, no, t or tr, mask, nm)
        agent3.update()
        tot_r += r
        obs, mask = no, nm
        if t or tr:
            break
    rewards.append(tot_r)
    covered_hist.append(inf.get("covered", 0))
    if inf.get("solved"):
        solved_count += 1

print(f"  Episodes: 500")
print(f"  Solved  : {solved_count}")
print(f"  Mean reward (last 100): {np.mean(rewards[-100:]):.2f}")
print(f"  Mean covered (last 100): {np.mean(covered_hist[-100:]):.1f}/40")

# Save a checkpoint
os.makedirs("demo_ckpt", exist_ok=True)
agent3.save("demo_ckpt/demo.pt")
print("  Checkpoint saved → demo_ckpt/demo.pt")
print("  ✓ Demo training passed\n")

print("All tests passed!")
