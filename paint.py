#!/usr/bin/env python3
"""Parse a trainer_state.json and plot reward vs step.

Usage examples:
  python paint.py --json Qwen2.5-VL-3B-Click/checkpoint-1520/trainer_state.json
  python paint.py --json trainer_state.json --out reward_plot.png --csv out.csv

The script looks for a top-level `log_history` list (as produced by HuggingFace Trainer)
and extracts entries that contain a numeric `reward` (or any key containing 'reward').
It plots reward against the entry's `step` (or `global_step` if present).
"""
import argparse
import json
import os
from typing import List, Tuple
from datetime import datetime

import matplotlib.pyplot as plt


def extract_reward_steps(data: dict) -> List[Tuple[int, float]]:
    candidates = []

    # Try common place
    log_history = None
    if isinstance(data, dict) and 'log_history' in data:
        log_history = data['log_history']
    # If file is a list already
    if log_history is None and isinstance(data, list):
        log_history = data

    if not isinstance(log_history, list):
        raise ValueError('trainer_state.json does not contain a log_history list')

    for entry in log_history:
        if not isinstance(entry, dict):
            continue

        # determine step key
        step = None
        for k in ('step', 'global_step', 'logging_step'):
            if k in entry:
                try:
                    step = int(entry[k])
                    break
                except Exception:
                    step = None

        # find reward with safer priority:
        # 1) exact 'reward' key
        # 2) keys under 'rewards/...' ending with '/mean' (e.g. 'rewards/hybrid_label_confidence_reward/mean')
        # 3) metrics dict similarly
        # 4) fallback to any key containing 'reward' (least preferred)
        reward = None
        # 1) exact
        if 'reward' in entry:
            try:
                reward = float(entry['reward'])
            except Exception:
                reward = None

        # 2) rewards/.../mean pattern
        if reward is None:
            for k, v in entry.items():
                lk = k.lower()
                if lk.startswith('rewards/') and lk.endswith('/mean'):
                    try:
                        reward = float(v)
                        break
                    except Exception:
                        continue

        # 3) inside metrics dict
        if reward is None and 'metrics' in entry and isinstance(entry['metrics'], dict):
            # prefer exact 'reward' inside metrics
            if 'reward' in entry['metrics']:
                try:
                    reward = float(entry['metrics']['reward'])
                except Exception:
                    reward = None
            if reward is None:
                for k, v in entry['metrics'].items():
                    lk = k.lower()
                    if lk.startswith('rewards/') and lk.endswith('/mean') or 'reward' == lk:
                        try:
                            reward = float(v)
                            break
                        except Exception:
                            continue

        # 4) fallback: any key containing 'reward'
        if reward is None:
            for k, v in entry.items():
                lk = k.lower()
                if 'reward' in lk:
                    try:
                        reward = float(v)
                        break
                    except Exception:
                        continue

        if reward is None:
            continue

        # if no explicit step, some logs carry 'step' elsewhere; skip if absent
        if step is None:
            # try to use 'epoch' as coarse x-axis
            if 'epoch' in entry:
                try:
                    step = int(entry['epoch'])
                except Exception:
                    step = None

        if step is None:
            # skip entries without a usable x coordinate
            continue

        candidates.append((step, reward))

    # sort by step
    candidates.sort(key=lambda x: x[0])
    return candidates


def extract_metric_steps(data: dict, metric_substr: str) -> List[Tuple[int, float]]:
    """Generic extractor: find entries where a key contains metric_substr (case-insensitive).

    Keeps the same step-detection logic as extract_reward_steps.
    """
    metric_substr = metric_substr.lower()
    candidates = []

    # Try common place
    log_history = None
    if isinstance(data, dict) and 'log_history' in data:
        log_history = data['log_history']
    if log_history is None and isinstance(data, list):
        log_history = data

    if not isinstance(log_history, list):
        return []

    for entry in log_history:
        if not isinstance(entry, dict):
            continue

        # determine step key
        step = None
        for k in ('step', 'global_step', 'logging_step'):
            if k in entry:
                try:
                    step = int(entry[k])
                    break
                except Exception:
                    step = None

        # find metric-like key
        val = None
        for k, v in entry.items():
            if metric_substr in k.lower():
                try:
                    val = float(v)
                    break
                except Exception:
                    continue

        # also accept inside metrics dict
        if val is None and 'metrics' in entry and isinstance(entry['metrics'], dict):
            for k, v in entry['metrics'].items():
                if metric_substr in k.lower():
                    try:
                        val = float(v)
                        break
                    except Exception:
                        continue

        if val is None:
            continue

        if step is None:
            if 'epoch' in entry:
                try:
                    step = int(entry['epoch'])
                except Exception:
                    step = None

        if step is None:
            continue

        candidates.append((step, val))

    candidates.sort(key=lambda x: x[0])
    return candidates


def plot_reward(points: List[Tuple[int, float]], out_path: str, title: str | None = None):
    if not points:
        raise ValueError('no reward points to plot')

    import numpy as _np

    steps, rewards = zip(*points)
    steps = _np.array(steps)
    rewards = _np.array(rewards, dtype=float)

    plt.figure(figsize=FIGSIZE)

    # raw
    if PLOT_RAW:
        plt.plot(steps, rewards, marker='o', linestyle='-', color='#1f77b4', linewidth=1.0, markersize=4, alpha=0.9, label='raw')

    # smooth (moving average)
    if PLOT_SMOOTH and SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
        window = int(SMOOTH_WINDOW)
        if window >= len(rewards):
            smooth = _np.convolve(rewards, _np.ones(len(rewards))/len(rewards), mode='valid')
            smooth_steps = steps[(window-1)//2: len(steps)- (window//2)] if len(steps) >= window else steps
        else:
            smooth = _np.convolve(rewards, _np.ones(window)/window, mode='valid')
            # align smooth x to center of window
            pad_left = (window-1)//2
            pad_right = window-1-pad_left
            smooth_steps = steps[pad_left: len(steps)-pad_right]

        plt.plot(smooth_steps, smooth, color='#ff7f0e', linewidth=2.0, label=f'moving_avg(w={window})')

    # annotate min/max on the raw rewards
    try:
        argmin = int(_np.argmin(rewards))
        argmax = int(_np.argmax(rewards))
        ymin = rewards[argmin]
        ymax = rewards[argmax]
        plt.scatter([steps[argmin]], [ymin], color='green', zorder=5)
        plt.scatter([steps[argmax]], [ymax], color='red', zorder=5)
        plt.annotate(f'min {ymin:.3f}', xy=(steps[argmin], ymin), xytext=(5, -15), textcoords='offset points', fontsize=8)
        plt.annotate(f'max {ymax:.3f}', xy=(steps[argmax], ymax), xytext=(5, 5), textcoords='offset points', fontsize=8)
    except Exception:
        pass

    plt.xlabel('step')
    plt.ylabel('reward')
    if title:
        plt.title(title)

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=False)

    # improve tick density and tighten loss axis when plotting loss
    try:
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))
        if title and 'loss' in title.lower():
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            # tighten y-limits around loss values for better resolution
            try:
                minr = float(rewards.min())
                maxr = float(rewards.max())
                delta = max((maxr - minr) * 0.2, 1e-8)
                plt.ylim(minr - delta, maxr + delta)
            except Exception:
                pass
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_path, dpi=OUT_DPI)
    print(f'Saved plot to {out_path}')


# --------------------- User constants ---------------------
# Modify these constants directly to point to your trainer_state.json
JSON_PATH = "Qwen2.5-VL-3B-Click-NewPlan3/checkpoint-796/trainer_state.json"
# Output plot path (PNG). If None, will be written next to the json file
OUT_PATH = ".output2.png"
# Optional CSV export path. Set to path or leave None to skip
CSV_PATH = ".output2.csv"
# Optional plot title
PLOT_TITLE = "Reward vs Logging Step"
# Visualization tuning constants (adjust for finer plots)
# If >1, plot a moving average (window in number of points)
SMOOTH_WINDOW = 3
# Whether to show raw points
PLOT_RAW = False
# Whether to show smoothed curve
PLOT_SMOOTH = True
# DPI and figure size for higher-resolution output
OUT_DPI = 300
FIGSIZE = (10, 4.5)
# Whether to also look for and plot 'loss' vs step
PLOT_LOSS = True
# Substring used to find loss-like keys in log entries
LOSS_KEY = 'loss'


def main_with_constants(json_path=JSON_PATH, out_path=OUT_PATH, csv_path=CSV_PATH, title=PLOT_TITLE):
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = extract_reward_steps(data)
    if not points:
        print('No (step, reward) pairs found in log_history. Will check for loss entries.')

    # Ensure output directory 'figure' exists in current working directory
    figure_dir = os.path.join(os.getcwd(), 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Naming convention: <metric>_vs_step_<timestamp>.<ext> placed in figure/
    def make_out_path(metric: str, ext: str = '.png') -> str:
        metric_clean = metric.lower().replace(' ', '_')
        return os.path.join(figure_dir, f"{metric_clean}_vs_step_{ts}{ext}")

    # Reward output path
    if out_path is None:
        out_path = make_out_path('reward', '.png')
    else:
        # respect provided extension but enforce metric-based name
        _, ext = os.path.splitext(os.path.basename(out_path))
        if not ext:
            ext = '.png'
        out_path = make_out_path('reward', ext)

    if points:
        plot_reward(points, out_path, title=title)

        if csv_path:
            # CSV naming: reward_vs_step_<ts>.csv (ignore user path, place in figure)
            csv_path = make_out_path('reward', '.csv')
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow(['step', 'reward'])
                for s, r in points:
                    writer.writerow([s, r])
            print(f'Saved CSV to {csv_path}')

    # Also attempt to extract and plot loss vs step if requested
    if PLOT_LOSS:
        loss_points = extract_metric_steps(data, LOSS_KEY)
        if loss_points:
            # derive loss output path from out_path
            # loss output path per naming convention
            loss_out = make_out_path('loss', '.png')
            plot_reward(loss_points, loss_out, title='Loss vs Step')

            if csv_path:
                loss_csv = make_out_path('loss', '.csv')
                import csv
                with open(loss_csv, 'w', newline='', encoding='utf-8') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(['step', 'loss'])
                    for s, v in loss_points:
                        writer.writerow([s, v])
                print(f'Saved loss CSV to {loss_csv}')


if __name__ == '__main__':
    main_with_constants()
