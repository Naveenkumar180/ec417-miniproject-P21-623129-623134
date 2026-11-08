"""Aggregate results/metrics.csv into the mandatory results table and make figures.

  python -m src.evaluate

Outputs: results/results_table.md, results/per_class_recall.md and
results/figures/{confusion_<model>.png, per_class_recall.png}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data import CLASS_NAMES

ORDER = [("majority", "Trivial baseline (majority class)"),
         ("rf", "Baseline: random forest (raw 187 samples)"),
         ("lstm", "Main: LSTM"),
         ("bilstm", "Main: BiLSTM"),
         ("cnnlstm", "Ablation: CNN-LSTM (no class weights)"),
         ("bilstm_w", "Ablation: BiLSTM + class weights"),
         ("cnnlstm_w", "+ improvement: CNN-LSTM + class weights")]


def ms(df, model, metric, fmt="{:.3f} ± {:.3f}"):
    v = df[(df.model == model) & (df.metric == metric)].value
    return "–" if v.empty else fmt.format(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    res = Path(a.results_dir)
    df = pd.read_csv(res / "metrics.csv")
    (res / "figures").mkdir(parents=True, exist_ok=True)
    present = [(m, lbl) for m, lbl in ORDER if m in set(df.model)]

    # ---- main table (test-set numbers, mean ± std over seeds) ----
    lines = ["| Model | Test macro-F1 | Test accuracy | Params | Train time (s) | Seeds |",
             "|---|---|---|---|---|---|"]
    for m, lbl in present:
        n = df[(df.model == m) & (df.metric == "test_macro_f1")].seed.nunique()
        n_par = int(df[(df.model == m) & (df.metric == "params")].value.mean())
        lines.append(f"| {lbl} | {ms(df, m, 'test_macro_f1')} | {ms(df, m, 'test_accuracy')} | "
                     f"{n_par:,} | {ms(df, m, 'train_time_s', '{:.0f} ± {:.0f}')} | {n} |")
    (res / "results_table.md").write_text("\n".join(lines) + "\n")

    # ---- per-class recall table ----
    lines = ["| Model | " + " | ".join(f"recall {c}" for c in CLASS_NAMES) + " |",
             "|---|" + "---|" * len(CLASS_NAMES)]
    for m, lbl in present:
        lines.append(f"| {lbl} | " + " | ".join(ms(df, m, f"test_recall_{c}") for c in CLASS_NAMES) + " |")
    (res / "per_class_recall.md").write_text("\n".join(lines) + "\n")

    # ---- per-class recall bar chart ----
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(present), 1)
    for i, (m, lbl) in enumerate(present):
        means = [df[(df.model == m) & (df.metric == f"test_recall_{c}")].value.mean() for c in CLASS_NAMES]
        errs = [df[(df.model == m) & (df.metric == f"test_recall_{c}")].value.std(ddof=0) for c in CLASS_NAMES]
        ax.bar(np.arange(5) + i * width, means, width, yerr=errs, label=m)
    ax.set_xticks(np.arange(5) + 0.4 - width / 2)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("test recall")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Per-class recall (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(res / "figures" / "per_class_recall.png", dpi=150)
    plt.close(fig)

    # ---- confusion matrices (seed 0 predictions) ----
    for m, _ in present:
        p = res / "preds" / f"{m}_seed0.npz"
        if not p.exists():
            continue
        d = np.load(p)
        cm = confusion_matrix(d["y_true"], d["y_pred"], labels=range(5))
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{m} (seed 0, test)")
        fig.tight_layout()
        fig.savefig(res / "figures" / f"confusion_{m}.png", dpi=150)
        plt.close(fig)
    print((res / "results_table.md").read_text())
    print((res / "per_class_recall.md").read_text())


if __name__ == "__main__":
    main()
