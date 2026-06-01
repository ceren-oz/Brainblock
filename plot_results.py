"""
Plot learning curves from training logs.

Usage:
    python plot_results.py --log_dirs runs/sac_dense_seed0/log.csv \
                                      runs/sac_dense_seed1/log.csv \
                                      runs/sac_dense_seed2/log.csv \
                                      runs/sac_dense_seed3/log.csv \
                                      runs/sac_dense_seed4/log.csv \
                           --out_dir plots/dense_all_seeds
"""

import argparse
import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log_dirs",
        nargs="+",
        required=True,
        help="Paths to log.csv files",
    )
    p.add_argument(
        "--window",
        type=int,
        default=200,
        help="Smoothing window for rolling average",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="plots",
    )
    return p.parse_args()


def load_log(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    k: float(v) if v != "" else float("nan")
                    for k, v in row.items()
                }
            )
    return rows


def smooth(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_metric(
    ax,
    episodes_list,
    values_list,
    labels,
    window,
    title,
    ylabel,
    color_list=None,
):
    if color_list is None:
        color_list = plt.cm.tab10.colors

    for eps, vals, lbl, col in zip(
        episodes_list,
        values_list,
        labels,
        color_list,
    ):
        if len(vals) < window:
            ax.plot(
                eps,
                vals,
                label=lbl,
                color=col,
                linewidth=1.5,
            )
        else:
            sv = smooth(vals, window)
            se = eps[window - 1 :]
            ax.plot(
                se,
                sv,
                label=lbl,
                color=col,
                linewidth=1.5,
            )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_episodes = []
    all_rewards = []
    all_lengths = []
    all_covered = []
    all_solved = []
    all_inv_rates = []
    labels = []

    for path in args.log_dirs:
        rows = load_log(path)

        eps = np.array([r["episode"] for r in rows])

        all_episodes.append(eps)
        all_rewards.append(np.array([r["reward"] for r in rows]))
        all_lengths.append(np.array([r["ep_len"] for r in rows]))
        all_covered.append(np.array([r["covered"] for r in rows]))
        all_solved.append(np.array([r["solved"] for r in rows]))
        all_inv_rates.append(
            np.array(
                [
                    float(r.get("invalid_rate", 0) or 0)
                    for r in rows
                ]
            )
        )

        # Use parent directory name as label
        labels.append(os.path.basename(os.path.dirname(path)))

    # --------------------------------------------------
    # Main training curves
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "BrainBlock SAC Training Curves",
        fontsize=13,
        fontweight="bold",
    )

    plot_metric(
        axes[0, 0],
        all_episodes,
        all_rewards,
        labels,
        args.window,
        "Total Reward vs Episode",
        "Episodic Return",
    )

    plot_metric(
        axes[0, 1],
        all_episodes,
        all_covered,
        labels,
        args.window,
        "Covered Cells vs Episode",
        "Covered Cells (out of 40)",
    )

    plot_metric(
        axes[1, 0],
        all_episodes,
        all_lengths,
        labels,
        args.window,
        "Episode Length vs Episode",
        "Steps per Episode",
    )

    plot_metric(
        axes[1, 1],
        all_episodes,
        all_solved,
        labels,
        args.window,
        "Success Rate vs Episode",
        "Success Rate (rolling mean)",
    )

    plt.tight_layout()

    fig_path = os.path.join(
        args.out_dir,
        "training_curves.png",
    )

    plt.savefig(
        fig_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()
    print(f"Saved → {fig_path}")

    # --------------------------------------------------
    # Invalid-action rate plot
    # --------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    plot_metric(
        ax2,
        all_episodes,
        all_inv_rates,
        labels,
        args.window,
        "Invalid-Action Rate vs Episode",
        "Invalid-Action Rate",
    )

    fig2.tight_layout()

    inv_path = os.path.join(
        args.out_dir,
        "invalid_action_rate.png",
    )

    plt.savefig(
        inv_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()
    print(f"Saved → {inv_path}")


if __name__ == "__main__":
    main()