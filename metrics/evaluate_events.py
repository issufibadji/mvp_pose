from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_events(path: Path):
    events = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        t_key = 't_sec' if 't_sec' in reader.fieldnames else 't'
        for row in reader:
            events.append((float(row[t_key]), row['event']))
    return events


def evaluate(pred, gt, tol: float = 0.6):
    gt_used = [False] * len(gt)
    counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for t_pred, name in pred:
        idx = None
        min_dt = tol
        for i, (t_gt, n_gt) in enumerate(gt):
            if gt_used[i] or n_gt != name:
                continue
            dt = abs(t_pred - t_gt)
            if dt <= min_dt:
                min_dt = dt
                idx = i
        if idx is not None:
            gt_used[idx] = True
            counts[name]['tp'] += 1
        else:
            counts[name]['fp'] += 1
    for i, (t_gt, name) in enumerate(gt):
        if not gt_used[i]:
            counts[name]['fn'] += 1
    metrics = {}
    for name, c in counts.items():
        tp, fp, fn = c['tp'], c['fp'], c['fn']
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        metrics[name] = (tp, fp, fn, prec, rec, f1)
    return metrics


def main():
    ap = argparse.ArgumentParser(description='Evaluate gesture events against ground truth.')
    ap.add_argument('pred', type=Path, help='Predicted events.csv')
    ap.add_argument('gt', type=Path, help='Ground truth events_gt.csv')
    ap.add_argument('--tol', type=float, default=0.6, help='Time tolerance in seconds')
    args = ap.parse_args()

    pred = load_events(args.pred)
    gt = load_events(args.gt)
    metrics = evaluate(pred, gt, args.tol)
    for name, (tp, fp, fn, prec, rec, f1) in metrics.items():
        print(f'{name}: TP={tp} FP={fp} FN={fn} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}')


if __name__ == '__main__':
    main()
