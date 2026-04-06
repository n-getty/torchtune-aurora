#!/usr/bin/env python3
"""Plot GRPO learning curves from DiskLogger output.

Usage:
    python recipes/dev/plot_learning_curves.py results/qwen32b_learning_run/logs/
    python recipes/dev/plot_learning_curves.py results/qwen32b_learning_run/logs/ results/gemma4_learning_run/logs/ --labels "Qwen3-32B" "Gemma4-31B"
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log_file(log_path: str) -> dict[str, list[tuple[int, float]]]:
    """Parse a DiskLogger txt file into {metric_name: [(step, value), ...]}."""
    metrics: dict[str, list[tuple[int, float]]] = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("Step"):
                continue
            # Format: "Step N | key1:val1 key2:val2 ..."
            match = re.match(r"Step\s+(\d+)\s*\|\s*(.*)", line)
            if not match:
                continue
            step = int(match.group(1))
            pairs = match.group(2).split()
            for pair in pairs:
                if ":" not in pair:
                    continue
                key, val_str = pair.split(":", 1)
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append((step, val))
    return metrics


def find_log_file(log_dir: str) -> str:
    """Find the DiskLogger output file in a log directory."""
    log_dir = Path(log_dir)
    # DiskLogger creates files like log_TIMESTAMP.txt
    candidates = sorted(log_dir.glob("*.txt"), key=os.path.getmtime, reverse=True)
    if candidates:
        return str(candidates[0])
    # Also check for .jsonl
    candidates = sorted(log_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(f"No log files found in {log_dir}")


def plot_learning_curves(
    log_dirs: list[str],
    labels: list[str],
    output_path: str = "learning_curves.png",
):
    """Plot training and eval metrics from one or more runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO Learning Curves — GSM8K", fontsize=14, fontweight="bold")

    plot_configs = [
        ("successes", "Train Success Rate", axes[0, 0]),
        ("eval/successes", "Eval Success Rate (held-out)", axes[0, 1]),
        ("rewards", "Train Rewards", axes[1, 0]),
        ("loss", "Loss", axes[1, 1]),
    ]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for run_idx, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        try:
            log_file = find_log_file(log_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        metrics = parse_log_file(log_file)
        color = colors[run_idx % len(colors)]

        for metric_key, title, ax in plot_configs:
            if metric_key in metrics:
                steps, values = zip(*metrics[metric_key])
                ax.plot(steps, values, label=label, color=color, linewidth=1.5,
                        marker="o" if len(steps) < 20 else None, markersize=4)
                ax.set_title(title)
                ax.set_xlabel("Step")
                ax.set_ylabel(metric_key.split("/")[-1])
                ax.legend()
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO learning curves")
    parser.add_argument("log_dirs", nargs="+", help="Log directories to plot")
    parser.add_argument("--labels", nargs="+", help="Labels for each run")
    parser.add_argument("-o", "--output", default="learning_curves.png",
                        help="Output PNG path")
    args = parser.parse_args()

    labels = args.labels or [Path(d).parent.name for d in args.log_dirs]
    if len(labels) < len(args.log_dirs):
        labels.extend([f"run_{i}" for i in range(len(labels), len(args.log_dirs))])

    plot_learning_curves(args.log_dirs, labels, args.output)


if __name__ == "__main__":
    main()
