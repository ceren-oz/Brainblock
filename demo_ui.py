import time
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

print("Matplotlib backend:", matplotlib.get_backend())

from environment import BrainBlockEnv, W, H
from sac_agent import SACAgent
from evaluate import PIECE_COLORS


# =========================
# CONSTANTS
# =========================
DETERMINISTIC = True
MAX_QUEUE_SHOW = 10

EMPTY_COLOR = "#FFFFFF"
GRID_COLOR = "#DDDDDD"


# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="runs/sac_dense_seed0/checkpoints/final.pt",
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Simulation speed multiplier (higher = faster)"
    )

    return parser.parse_args()


# =========================
# DEMO UI
# =========================
class DemoUI:
    def __init__(self, agent, env, speed):
        self.agent = agent
        self.env = env
        self.speed = speed

        self.obs, _ = env.reset(seed=42)
        self.mask = env.get_valid_action_mask()

        self.done = False

        # cumulative reward tracker
        self.episode_reward = 0.0

        # cell memory
        self.cell_colors = [
            [EMPTY_COLOR for _ in range(W)]
            for _ in range(H)
        ]
        self.prev_board = np.zeros((H, W), dtype=int)

        # matplotlib GUI
        plt.ion()

        self.fig = plt.figure(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.ax_board = self.fig.add_subplot(1, 2, 1)
        self.ax_queue = self.fig.add_subplot(1, 2, 2)

    # =========================
    # STEP
    # =========================
    def step(self):
        if self.done:
            return

        piece = self.env.queue[0] if len(self.env.queue) > 0 else None

        action = self.agent.select_action(
            self.obs,
            self.mask,
            deterministic=DETERMINISTIC
        )

        self.obs, reward, terminated, truncated, info = self.env.step(action)

        self.mask = self.env.get_valid_action_mask()
        self.done = terminated or truncated

        self.episode_reward += reward

        self._update_cell_colors(piece)
        self.render(reward, info)

        time.sleep(0.5 / self.speed)

    # =========================
    # UPDATE COLORS
    # =========================
    def _update_cell_colors(self, piece):
        board = self.env.board
        diff = board - self.prev_board

        if piece is not None:
            color = PIECE_COLORS.get(piece, "#999999")

            for r in range(H):
                for c in range(W):
                    if diff[r, c] == 1:
                        self.cell_colors[r][c] = color

        self.prev_board = board.copy()

    # =========================
    # BOARD
    # =========================
    def render_board(self):
        self.ax_board.clear()

        self.ax_board.set_xlim(0, W)
        self.ax_board.set_ylim(0, H)
        self.ax_board.set_aspect("equal")
        self.ax_board.invert_yaxis()
        self.ax_board.set_title("BrainBlock Board")

        for r in range(H):
            for c in range(W):
                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.8,
                    edgecolor=GRID_COLOR,
                    facecolor=self.cell_colors[r][c]
                )
                self.ax_board.add_patch(rect)

        self.ax_board.axis("off")

    # =========================
    # QUEUE
    # =========================
    def render_queue(self):
        self.ax_queue.clear()
        self.ax_queue.set_title("Piece Queue")
        self.ax_queue.axis("off")

        queue = self.env.queue[:MAX_QUEUE_SHOW]

        for i, piece in enumerate(queue):
            y = 1 - i * 0.09
            is_current = (i == 0)

            color = PIECE_COLORS.get(piece, "#999999")

            self.ax_queue.text(
                0.1, y,
                ("> " if is_current else "  ") + piece,
                fontsize=14 if is_current else 12,
                fontweight="bold" if is_current else "normal",
                color=color
            )

        self.ax_queue.text(
            0.1, 0.05,
            f"Remaining: {len(self.env.queue)}",
            fontsize=10,
            color="black"
        )

    # =========================
    # RENDER
    # =========================
    def render(self, reward, info):
        self.render_board()
        self.render_queue()

        self.fig.suptitle(
            f"Step Reward: {reward:.2f} | "
            f"Total Reward: {self.episode_reward:.2f} | "
            f"Step: {self.env.step_count} | "
            f"Covered: {info.get('covered', 0)}/40",
            fontsize=12
        )

        self.canvas.draw()
        self.canvas.flush_events()


# =========================
# RUN DEMO
# =========================
def run_demo(args):
    env = BrainBlockEnv(reward_fn="dense")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        device=device
    )

    agent.load(args.ckpt)

    ui = DemoUI(agent, env, speed=args.speed)

    print(f"Starting RL Demo... ckpt={args.ckpt}, speed={args.speed}")

    try:
        while True:
            if ui.done:
                time.sleep(1.0)

                ui.obs, _ = env.reset(seed=np.random.randint(0, 9999))
                ui.mask = env.get_valid_action_mask()
                ui.done = False

                ui.cell_colors = [[EMPTY_COLOR for _ in range(W)] for _ in range(H)]
                ui.prev_board = np.zeros((H, W), dtype=int)
                ui.episode_reward = 0.0

            ui.step()

    except KeyboardInterrupt:
        print("\nDemo stopped.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()
    run_demo(args)