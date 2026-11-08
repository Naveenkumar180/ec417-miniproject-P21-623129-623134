"""Train ONE configuration with ONE seed and append its metrics to results/metrics.csv.

Examples
  python -m src.train --model majority --seed 0
  python -m src.train --model rf       --seed 0
  python -m src.train --model lstm     --seed 0
  python -m src.train --model bilstm   --seed 0
  python -m src.train --model cnnlstm  --seed 0                # CNN-LSTM, no weighting (ablation)
  python -m src.train --model cnnlstm  --weighted --seed 0     # full improvement

Model selection (best epoch) uses VALIDATION macro-F1 only. The test set is
scored once, after training has finished.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data import CLASS_NAMES, class_weights, make_splits
from src.models import build_rf, build_torch_model, count_params


def metrics_rows(name, seed, split, y_true, y_pred):
    rec = recall_score(y_true, y_pred, labels=range(5), average=None, zero_division=0)
    rows = [(name, seed, f"{split}_macro_f1", f1_score(y_true, y_pred, average="macro", zero_division=0)),
            (name, seed, f"{split}_accuracy", accuracy_score(y_true, y_pred))]
    rows += [(name, seed, f"{split}_recall_{CLASS_NAMES[i]}", rec[i]) for i in range(5)]
    return rows


def append_metrics(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "seed", "metric", "value"])
        w.writerows(rows)


def run_torch(args, S, name):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_torch_model(args.model).to(dev)
    n_params = count_params(model)

    w = torch.tensor(class_weights(S["y_train"]), device=dev) if args.weighted else None
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    T = lambda a, dt: torch.tensor(a, dtype=dt)
    tr = DataLoader(TensorDataset(T(S["X_train"], torch.float32), T(S["y_train"], torch.long)),
                    batch_size=args.batch_size, shuffle=True)

    @torch.no_grad()
    def predict(X):
        model.eval()
        out = []
        for i in range(0, len(X), 1024):
            out.append(model(T(X[i:i + 1024], torch.float32).to(dev)).argmax(1).cpu())
        return torch.cat(out).numpy()

    best_f1, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        f1 = f1_score(S["y_val"], predict(S["X_val"]), average="macro")
        print(f"[{name} s{args.seed}] epoch {ep + 1:02d} val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    return predict(S["X_val"]), predict(S["X_test"]), train_time, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["majority", "rf", "lstm", "bilstm", "cnnlstm"])
    ap.add_argument("--weighted", action="store_true", help="class-weighted loss (torch models)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    name = args.model + ("_w" if args.weighted else "")
    S = make_splits(args.data_dir)

    if args.model == "majority":
        t0 = time.time()
        maj = np.bincount(S["y_train"]).argmax()
        pv, pt = np.full_like(S["y_val"], maj), np.full_like(S["y_test"], maj)
        train_time, n_params = time.time() - t0, 0
    elif args.model == "rf":
        clf = build_rf(args.seed)
        t0 = time.time()
        clf.fit(S["X_train"], S["y_train"])
        train_time, n_params = time.time() - t0, 0
        pv, pt = clf.predict(S["X_val"]), clf.predict(S["X_test"])
    else:
        pv, pt, train_time, n_params = run_torch(args, S, name)

    rows = metrics_rows(name, args.seed, "val", S["y_val"], pv)
    rows += metrics_rows(name, args.seed, "test", S["y_test"], pt)
    rows += [(name, args.seed, "train_time_s", train_time), (name, args.seed, "params", n_params)]
    res = Path(args.results_dir)
    append_metrics(res / "metrics.csv", rows)
    (res / "preds").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(res / "preds" / f"{name}_seed{args.seed}.npz", y_true=S["y_test"], y_pred=pt)
    f1 = [r[3] for r in rows if r[2] == "test_macro_f1"][0]
    print(f"{name} seed {args.seed}: test macro-F1 {f1:.4f}, train time {train_time:.1f}s, params {n_params}")


if __name__ == "__main__":
    main()
