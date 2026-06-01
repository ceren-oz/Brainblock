"""
GIF solution visualizer for BrainBlock.

Produces an animated GIF that shows:
  - The board state building up piece by piece
  - The piece queue on the right: upcoming pieces rendered at small scale,
    with already-placed pieces faded/greyed out
  - A header showing the current step and piece being placed

Usage (standalone):
    python visualize_gif.py --ckpt demo_ckpt/demo.pt --out_dir eval_out --n_solutions 3

Called from evaluate.py via:
    from visualize_gif import render_solution_gif
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

from environment import (
    BrainBlockEnv, PIECE_TYPES, PIECE_TRANSFORMS, INVENTORY, W, H
)

# -------------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------------

PIECE_COLORS = {
    "I": "#4C9BE8",
    "O": "#E8C44C",
    "L": "#4CE87A",
    "Z": "#E84C4C",
    "T": "#C44CE8",
}
EMPTY_COLOR   = "#2A2A3A"
GRID_COLOR    = "#3A3A4A"
BG_COLOR      = "#1A1A2E"
FADED_ALPHA   = 0.22   # alpha for already-placed queue items
ACTIVE_SCALE  = 1.0    # relative scale for the active (next) piece in queue
QUEUE_CELL    = 0.45   # cell size (in inches) for the mini piece preview

# How long each frame stays (milliseconds)
FRAME_DURATION_MS = 420
FINAL_HOLD_MS     = 1800   # last frame lingers longer


# -------------------------------------------------------------------------
# Helpers: draw a single mini-piece in an axes
# -------------------------------------------------------------------------

def _piece_bounding_box(cells):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), max(xs), min(ys), max(ys)


def _draw_mini_piece(ax, cells, color, alpha=1.0, cell_size=1.0):
    """Draw a tetromino shape centred in `ax`, using orientation 0 canonical cells."""
    min_x, max_x, min_y, max_y = _piece_bounding_box(cells)
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.2, h + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")
    for (cx, cy) in cells:
        rect = mpatches.Rectangle(
            (cx - min_x, cy - min_y), 1, 1,
            linewidth=1.2,
            edgecolor="white" if alpha > 0.5 else "#555555",
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)


# -------------------------------------------------------------------------
# Core: render one frame of the animation
# -------------------------------------------------------------------------

def _render_frame(
    board: np.ndarray,
    cell_colors,           # H×W list of colour strings
    full_queue: list,      # original full piece queue (length 10)
    placed_count: int,     # how many pieces have been placed so far
    step_label: str,
    fig=None,
    axes=None,
):
    """
    Returns a (fig, (ax_board, ax_queue)) tuple.
    Re-uses `fig` and `axes` if provided (faster re-draw).
    """
    if fig is None:
        fig = plt.figure(figsize=(11, 5.5), facecolor=BG_COLOR)
        gs  = gridspec.GridSpec(
            1, 2,
            width_ratios=[8, 2.8],
            left=0.04, right=0.97, top=0.90, bottom=0.06,
            wspace=0.18,
        )
        ax_board = fig.add_subplot(gs[0])
        ax_queue = fig.add_subplot(gs[1])
    else:
        ax_board, ax_queue = axes
        ax_board.cla()
        ax_queue.cla()

    # ---- Board ----
    ax_board.set_facecolor(BG_COLOR)
    ax_board.set_xlim(0, W)
    ax_board.set_ylim(0, H)
    ax_board.set_aspect("equal")
    ax_board.invert_yaxis()
    ax_board.axis("off")

    # Draw grid background
    for r in range(H):
        for c in range(W):
            bg = mpatches.Rectangle(
                (c, r), 1, 1,
                linewidth=0.5,
                edgecolor=GRID_COLOR,
                facecolor=EMPTY_COLOR,
            )
            ax_board.add_patch(bg)

    # Draw filled cells
    for r in range(H):
        for c in range(W):
            col = cell_colors[r][c]
            if col != EMPTY_COLOR:
                rect = mpatches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.8,
                    edgecolor="#FFFFFF33",
                    facecolor=col,
                )
                ax_board.add_patch(rect)

    ax_board.set_title(step_label, color="white", fontsize=11,
                       pad=6, fontweight="bold")

    # ---- Queue panel ----
    ax_queue.set_facecolor(BG_COLOR)
    ax_queue.axis("off")

    n_total = len(full_queue)
    label_y_top = 0.96
    ax_queue.text(
        0.5, label_y_top, "Queue",
        transform=ax_queue.transAxes,
        ha="center", va="top",
        color="white", fontsize=10, fontweight="bold",
    )

    slot_height = (label_y_top - 0.02) / n_total
    for i, piece_type in enumerate(full_queue):
        placed = i < placed_count
        active = i == placed_count  # next to be placed

        cells = PIECE_TRANSFORMS[piece_type][0]  # canonical orientation
        color = PIECE_COLORS[piece_type]
        alpha = FADED_ALPHA if placed else 1.0

        # Slot rect (visual highlight for active)
        slot_y0 = label_y_top - (i + 1) * slot_height
        slot_y1 = slot_y0 + slot_height

        if active:
            highlight = FancyBboxPatch(
                (0.06, slot_y0 + 0.005), 0.88, slot_height - 0.01,
                boxstyle="round,pad=0.01",
                linewidth=1.5,
                edgecolor="#FFFFFF88",
                facecolor="#FFFFFF18",
                transform=ax_queue.transAxes,
                zorder=0,
            )
            ax_queue.add_patch(highlight)

        # Mini piece axes inset
        min_x, max_x, min_y, max_y = _piece_bounding_box(cells)
        pw = max_x - min_x + 1
        ph = max_y - min_y + 1

        margin_x = 0.10
        avail_w  = 1.0 - 2 * margin_x
        piece_w  = avail_w * 0.7
        piece_h  = slot_height * 0.72

        cx = margin_x + avail_w * 0.10
        cy = slot_y0 + slot_height * 0.14

        inset = ax_queue.inset_axes(
            [cx, cy, piece_w, piece_h],
            transform=ax_queue.transAxes,
        )
        _draw_mini_piece(inset, cells, color, alpha=alpha)

        # Piece label
        label_x = margin_x + avail_w * 0.85
        label_y = slot_y0 + slot_height * 0.5
        ax_queue.text(
            label_x, label_y,
            piece_type,
            transform=ax_queue.transAxes,
            ha="center", va="center",
            color=color if not placed else "#555555",
            fontsize=9 + (2 if active else 0),
            fontweight="bold" if active else "normal",
            alpha=alpha if placed else 1.0,
        )

        # Strikethrough line for placed pieces
        if placed:
            ax_queue.plot(
                [0.12, 0.88],
                [(slot_y0 + slot_y1) / 2, (slot_y0 + slot_y1) / 2],
                transform=ax_queue.transAxes,
                color="#555555", linewidth=1.0, alpha=0.7,
            )

    return fig, (ax_board, ax_queue)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def render_solution_gif(
    board_history: list,
    piece_sequence: list,
    full_queue: list,
    title: str,
    save_path: str,
    frame_duration_ms: int = FRAME_DURATION_MS,
    final_hold_ms: int = FINAL_HOLD_MS,
):
    """
    Render an animated GIF of one BrainBlock solution.

    Parameters
    ----------
    board_history   : list of (H,W) np.ndarray  — board after each placement
    piece_sequence  : list of str               — piece type placed at each step
    full_queue      : list of str               — original queue before any placement
    title           : str                       — used in header / filename
    save_path       : str                       — output path (must end in .gif)
    frame_duration_ms : int
    final_hold_ms   : int
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for GIF export.  pip install Pillow")

    frames_pil = []
    durations  = []

    # We'll render the initial empty board + one frame per placement
    fig  = None
    axes = None

    cell_colors = [[EMPTY_COLOR] * W for _ in range(H)]
    prev_board  = np.zeros((H, W), dtype=int)

    # Frame 0: empty board
    label = f"{title} — Step 0 / {len(piece_sequence)}"
    fig, axes = _render_frame(
        prev_board, cell_colors, full_queue, 0, label, fig, axes
    )
    frames_pil.append(_fig_to_pil(fig))
    durations.append(frame_duration_ms)

    for step_idx, (board, piece) in enumerate(zip(board_history, piece_sequence)):
        # Update cell colours for newly placed cells
        diff = board - prev_board
        color = PIECE_COLORS.get(piece, "#888888")
        for r in range(H):
            for c in range(W):
                if diff[r, c] == 1:
                    cell_colors[r][c] = color
        prev_board = board.copy()

        placed_count = step_idx + 1
        label = (
            f"{title} — Step {placed_count} / {len(piece_sequence)}  "
            f"| Placed: {piece}"
        )
        fig, axes = _render_frame(
            board, cell_colors, full_queue, placed_count, label, fig, axes
        )
        is_last = step_idx == len(board_history) - 1
        frames_pil.append(_fig_to_pil(fig))
        durations.append(final_hold_ms if is_last else frame_duration_ms)

    plt.close(fig)

    # Save GIF
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    frames_pil[0].save(
        save_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"  GIF saved → {save_path}  ({len(frames_pil)} frames)")


def _fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL Image without writing to disk."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).convert("RGBA").convert("RGB")


# -------------------------------------------------------------------------
# Standalone CLI
# -------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Generate solution GIFs for BrainBlock")
    p.add_argument("--ckpt",         type=str,  required=True)
    p.add_argument("--reward_fn",    type=str,  default="dense",
                   choices=["sparse", "dense"])
    p.add_argument("--n_solutions",  type=int,  default=3)
    p.add_argument("--out_dir",      type=str,  default="eval_out/gifs")
    p.add_argument("--frame_ms",     type=int,  default=FRAME_DURATION_MS)
    p.add_argument("--final_hold_ms",type=int,  default=FINAL_HOLD_MS)
    p.add_argument("--seed_start",   type=int,  default=0)
    p.add_argument("--max_attempts", type=int,  default=200,
                   help="Max episodes to attempt before giving up")
    return p.parse_args()


def main():
    args = _parse_args()

    import torch
    from sac_agent import SACAgent
    from evaluate import rollout_with_history

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env    = BrainBlockEnv(reward_fn=args.reward_fn)
    obs_dim  = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    agent = SACAgent(obs_dim=obs_dim, n_actions=n_actions, device=device)
    agent.load(args.ckpt)
    print(f"Loaded: {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)
    collected = 0
    attempt   = 0

    while collected < args.n_solutions and attempt < args.max_attempts:
        seed = args.seed_start + attempt
        attempt += 1

        # Capture the queue before reset modifies it
        env.reset(seed=seed)
        original_queue = list(env.queue)

        solved, hist, pseq, _ = rollout_with_history(env, agent, seed=seed)
        if not solved:
            continue

        collected += 1
        save_path = os.path.join(args.out_dir, f"solution_{collected:02d}.gif")
        render_solution_gif(
            hist, pseq, original_queue,
            title=f"Solution #{collected}",
            save_path=save_path,
            frame_duration_ms=args.frame_ms,
            final_hold_ms=args.final_hold_ms,
        )

    print(f"\nDone. {collected} GIF(s) saved to {args.out_dir}/")


if __name__ == "__main__":
    main()