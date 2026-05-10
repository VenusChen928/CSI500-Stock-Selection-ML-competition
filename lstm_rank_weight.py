"""
Sequence model that learns future rank and soft portfolio weights directly.

This is an experimental challenger to the tuned XGBoost route.  It uses a small
LSTM over recent per-stock feature sequences, then selects a rank-weighted
portfolio with validation-window scoring.
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

# Keep Torch/BLAS single-threaded by default.  The small LSTM does not benefit
# much from aggressive threading, while macOS Accelerate/OpenMP oversubscription
# has repeatedly produced uninterruptible worker processes in long backtests.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baseline_xgboost import MAX_WEIGHT, MIN_STOCKS
from features import (
    CORE_FEATURE_COLUMNS,
    CANDIDATE_FEATURE_GROUPS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
)
from score_submission import score_window

DATA_DIR = Path(__file__).parent / "data"


def _device() -> torch.device:
    override = os.environ.get("MLCOMP_DEVICE")
    if override:
        return torch.device(override)
    # CPU is the safer default for repeatable competition runs.  MPS can still
    # be requested explicitly with MLCOMP_DEVICE=mps for one-off experiments.
    return torch.device("cpu")


DEVICE = _device()
SEED = 42
SEQ_LEN = 30
BATCH_SIZE = 768
EPOCHS = 8
VAL_DAYS = 15
EMBARGO_DAYS = 5
LOOKBACK_DAYS = 475

SEQUENCE_FEATURES = CORE_FEATURE_COLUMNS + [
    "ret_3d",
    "mom_accel_5_20",
    "reversal_3d",
    "vol_5d",
    "vol_ratio_5_20",
    "amount_z_20d",
    "turnover_z_20d",
    "intraday_ret",
    "overnight_ret",
    "close_pos_20d",
    "drawdown_20d",
]


@dataclass(frozen=True)
class Policy:
    top_k: int
    temperature: float
    rank_blend: float


@dataclass
class Sample:
    date: pd.Timestamp
    stock_code: str
    seq: np.ndarray
    rank_target: float
    weight_target: float


class Normalizer:
    def __init__(self, records: list[Sample]):
        arr = np.concatenate([r.seq for r in records], axis=0)
        self.mean = arr.mean(axis=0)
        self.std = np.clip(arr.std(axis=0), 1e-6, None)

    def apply(self, records: list[Sample]) -> list[Sample]:
        return [
            Sample(
                date=r.date,
                stock_code=r.stock_code,
                seq=((r.seq - self.mean) / self.std).astype(np.float32),
                rank_target=r.rank_target,
                weight_target=r.weight_target,
            )
            for r in records
        ]


class SequenceDataset(Dataset):
    def __init__(self, records: list[Sample]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return (
            torch.tensor(r.seq, dtype=torch.float32),
            torch.tensor(r.rank_target, dtype=torch.float32),
            torch.tensor(r.weight_target, dtype=torch.float32),
        )


class LSTMRankWeight(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.rank_head = nn.Linear(hidden, 1)
        self.weight_head = nn.Linear(hidden, 1)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        h = self.head(out[:, -1, :])
        return self.rank_head(h).squeeze(-1), self.weight_head(h).squeeze(-1)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(int(os.environ.get("MLCOMP_TORCH_THREADS", "1")))
        torch.set_num_interop_threads(int(os.environ.get("MLCOMP_TORCH_INTEROP_THREADS", "1")))
    except RuntimeError:
        pass
    try:
        torch.backends.mkldnn.enabled = False
    except AttributeError:
        pass


def _target_columns(horizon: int) -> tuple[str, str, str, str]:
    suffix = f"{horizon}d"
    return (
        f"idx_target_{suffix}",
        f"future_excess_{suffix}",
        f"target_rank_{suffix}",
        f"target_weight_{suffix}",
    )


def add_direct_targets(
    panel: pd.DataFrame,
    index_df: pd.DataFrame,
    horizon: int = FORWARD_HORIZON,
) -> pd.DataFrame:
    idx = index_df[["date", "close"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx_col, excess_col, rank_col, weight_col = _target_columns(horizon)
    stock_target_col = f"target_{horizon}d"
    if stock_target_col not in panel.columns:
        panel = panel.sort_values(["stock_code", "date"]).copy()
        panel[stock_target_col] = panel.groupby("stock_code")["close"].shift(-horizon) / panel["close"] - 1.0
    idx[idx_col] = idx["close"].shift(-horizon) / idx["close"] - 1.0
    if idx_col in panel.columns:
        out = panel.copy()
    else:
        out = panel.merge(idx[["date", idx_col]], on="date", how="left")
    out[excess_col] = out[stock_target_col] - out[idx_col]
    out[rank_col] = out.groupby("date")[excess_col].rank(method="average", pct=True)

    positive = out[excess_col].clip(lower=0.0)
    top_mask = out[rank_col] >= 0.80
    raw = positive.where(top_mask, 0.0)
    denom = raw.groupby(out["date"]).transform("sum")
    rank_floor = (out[rank_col] - 0.80).clip(lower=0.0)
    floor_denom = rank_floor.groupby(out["date"]).transform("sum")
    out[weight_col] = np.where(
        denom > 0,
        raw / denom,
        np.where(floor_denom > 0, rank_floor / floor_denom, 0.0),
    )
    return out


def date_splits(train_pool: pd.DataFrame):
    dates = np.sort(train_pool["date"].unique())
    val_start = pd.Timestamp(dates[-VAL_DAYS])
    train_end = pd.Timestamp(dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    return (
        [pd.Timestamp(d) for d in dates if pd.Timestamp(d) <= train_end],
        [pd.Timestamp(d) for d in dates if pd.Timestamp(d) >= val_start],
        train_end,
        val_start,
    )


def build_records(
    panel: pd.DataFrame,
    dates: list[pd.Timestamp],
    target_horizon: int = FORWARD_HORIZON,
    require_targets: bool = True,
) -> list[Sample]:
    records: list[Sample] = []
    target_dates = {pd.Timestamp(d) for d in dates}
    _, _, rank_col, weight_col = _target_columns(target_horizon)
    panel = panel.sort_values(["stock_code", "date"]).reset_index(drop=True)
    for stock_code, g in panel.groupby("stock_code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        values = g[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
        dates_arr = [pd.Timestamp(d) for d in g["date"].to_list()]
        if require_targets:
            rank_target = g[rank_col].to_numpy(dtype=np.float32)
            weight_target = g[weight_col].to_numpy(dtype=np.float32)
        else:
            rank_target = np.zeros(len(g), dtype=np.float32)
            weight_target = np.zeros(len(g), dtype=np.float32)
        for i, d in enumerate(dates_arr):
            if d not in target_dates or i < SEQ_LEN - 1:
                continue
            seq = values[i - SEQ_LEN + 1 : i + 1]
            if not np.isfinite(seq).all():
                continue
            if require_targets and (
                not np.isfinite(rank_target[i]) or not np.isfinite(weight_target[i])
            ):
                continue
            records.append(
                Sample(
                    date=d,
                    stock_code=str(stock_code),
                    seq=seq,
                    rank_target=float(rank_target[i]),
                    weight_target=float(weight_target[i]),
                )
            )
    return records


def train_model(train_records: list[Sample], val_records: list[Sample]):
    set_seed()
    model = LSTMRankWeight(len(SEQUENCE_FEATURES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        SequenceDataset(train_records),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        SequenceDataset(val_records),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    best_state = None
    best_val = float("inf")
    wait = 0
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for seq, rank_t, weight_t in train_loader:
            seq = seq.to(DEVICE)
            rank_t = rank_t.to(DEVICE)
            weight_t = weight_t.to(DEVICE)
            rank_pred, weight_pred = model(seq)
            rank_loss = F.smooth_l1_loss(rank_pred, rank_t)
            weight_loss = F.mse_loss(torch.relu(weight_pred), torch.sqrt(weight_t))
            top_loss = (((rank_pred - rank_t) ** 2) * (1.0 + 2.0 * (rank_t > 0.80).float())).mean()
            sample_weight = 1.0 + 5.0 * torch.sqrt(weight_t)
            loss = (rank_loss + 0.6 * weight_loss + 0.20 * top_loss) * sample_weight.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq, rank_t, weight_t in val_loader:
                rank_pred, weight_pred = model(seq.to(DEVICE))
                rank_t = rank_t.to(DEVICE)
                weight_t = weight_t.to(DEVICE)
                rank_loss = F.smooth_l1_loss(rank_pred, rank_t)
                weight_loss = F.mse_loss(torch.relu(weight_pred), torch.sqrt(weight_t))
                val_losses.append(float((rank_loss + 0.6 * weight_loss).item()))
        val_loss = float(np.mean(val_losses))
        print(f"epoch {epoch + 1:02d} train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 2:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_records(model, records: list[Sample]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["date", "stock_code", "rank_pred", "weight_pred"])
    model.eval()
    loader = DataLoader(SequenceDataset(records), batch_size=BATCH_SIZE, shuffle=False)
    rank_preds = []
    weight_preds = []
    with torch.inference_mode():
        for seq, _, _ in loader:
            rank_pred, weight_pred = model(seq.to(DEVICE))
            rank_preds.append(rank_pred.cpu().numpy())
            weight_preds.append(weight_pred.cpu().numpy())
    return pd.DataFrame(
        {
            "date": [r.date for r in records],
            "stock_code": [r.stock_code for r in records],
            "rank_pred": np.concatenate(rank_preds),
            "weight_pred": np.concatenate(weight_preds),
        }
    )


def apply_cap(weights: pd.Series) -> pd.Series:
    w = weights[weights > 0].copy()
    if len(w) < MIN_STOCKS:
        raise ValueError("too few names")
    w = w / w.sum()
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def weights_from_predictions(pred: pd.DataFrame, as_of: pd.Timestamp, policy: Policy) -> pd.Series:
    d = pred[pred["date"] == as_of].copy()
    d["rank_pct"] = d["rank_pred"].rank(method="average", pct=True)
    d["weight_score"] = np.clip(d["weight_pred"], 0.0, None)
    d["raw_score"] = (
        policy.rank_blend * np.square(np.clip(d["rank_pct"] - 0.30, 0.0, None))
        + (1.0 - policy.rank_blend) * np.square(d["weight_score"])
    )
    d = d.sort_values("raw_score", ascending=False).head(policy.top_k)
    logits = d["raw_score"].to_numpy()
    if np.allclose(logits.sum(), 0.0):
        logits = d["rank_pct"].to_numpy()
    logits = logits / max(float(np.std(logits)), 1e-6)
    exp = np.exp((logits - logits.max()) / policy.temperature)
    w = pd.Series(exp / exp.sum(), index=d["stock_code"].astype(str))
    return apply_cap(w)


def select_policy(
    pred: pd.DataFrame,
    val_dates,
    trading_dates,
    prices,
    index_df,
    policy_horizon: int = FORWARD_HORIZON,
):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    windows = []
    for d in val_dates[::policy_horizon]:
        d = pd.Timestamp(d)
        idx = date_to_idx.get(d)
        if idx is None or idx + policy_horizon >= len(trading_dates):
            continue
        windows.append((d, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + policy_horizon])))
    min_windows = max(1, int(np.ceil(len(windows) * 0.8)))
    rows = []
    for top_k in (40, 50, 60, 80, 100):
        for temperature in (0.45, 0.65, 0.85, 1.10):
            for rank_blend in (0.50, 0.70, 0.90, 1.00):
                scores = []
                policy = Policy(top_k, temperature, rank_blend)
                for d, start, end in windows:
                    w = weights_from_predictions(pred, d, policy)
                    scores.append(
                        score_window(
                            w,
                            prices,
                            index_df,
                            start,
                            end,
                        )["excess_return"]
                    )
                if scores:
                    mean_score = float(np.mean(scores))
                    min_score = float(np.min(scores))
                    std_score = float(np.std(scores))
                    rows.append(
                        {
                            "top_k": top_k,
                            "temperature": temperature,
                            "rank_blend": rank_blend,
                            "n_windows": len(scores),
                            "required_windows": min_windows,
                            "mean_excess_return": mean_score,
                            "sum_excess_return": float(np.sum(scores)),
                            "min_excess_return": min_score,
                            "std_excess_return": std_score,
                            "utility_score": mean_score + min_score - 0.5 * std_score,
                        }
                    )
    table = pd.DataFrame(rows)
    eligible = table[table["n_windows"] >= min_windows].copy()
    if eligible.empty:
        eligible = table
    table = eligible.sort_values(
        ["utility_score", "mean_excess_return", "min_excess_return", "sum_excess_return"],
        ascending=False,
    )
    best = table.iloc[0]
    return Policy(int(best["top_k"]), float(best["temperature"]), float(best["rank_blend"])), table


def fit_lstm(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    policy_horizon: int = FORWARD_HORIZON,
    target_horizon: int = FORWARD_HORIZON,
):
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    min_date = as_of - pd.Timedelta(days=LOOKBACK_DAYS)
    prices = prices[prices["date"] >= min_date].copy()
    index_df = index_df[index_df["date"] >= min_date].copy()
    panel = add_direct_targets(build_features(prices, index_df), index_df, horizon=target_horizon)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - target_horizon)])
    _, _, rank_col, weight_col = _target_columns(target_horizon)
    usable = panel.dropna(subset=SEQUENCE_FEATURES + [rank_col, weight_col])
    train_pool = usable[usable["date"] <= train_cutoff].copy()
    train_dates, val_dates, train_end, val_start = date_splits(train_pool)
    train_records_raw = build_records(panel, train_dates, target_horizon=target_horizon)
    val_records_raw = build_records(panel, val_dates, target_horizon=target_horizon)
    normalizer = Normalizer(train_records_raw)
    train_records = normalizer.apply(train_records_raw)
    val_records = normalizer.apply(val_records_raw)
    print(f">> device={DEVICE} train_records={len(train_records):,} val_records={len(val_records):,}")
    print(f">> train_end={train_end.date()} val_start={val_start.date()}")
    model = train_model(train_records, val_records)
    val_pred = predict_records(model, val_records)
    policy, policy_table = select_policy(
        val_pred,
        val_dates,
        trading_dates,
        prices,
        index_df,
        policy_horizon=policy_horizon,
    )
    return {
        "panel": panel,
        "trading_dates": trading_dates,
        "normalizer": normalizer,
        "model": model,
        "policy": policy,
        "policy_table": policy_table,
        "prices": prices,
        "index_df": index_df,
        "target_horizon": target_horizon,
    }


def generate_submission(fit_result: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    records = build_records(
        fit_result["panel"],
        [as_of],
        target_horizon=fit_result.get("target_horizon", FORWARD_HORIZON),
        require_targets=False,
    )
    records = fit_result["normalizer"].apply(records)
    pred = predict_records(fit_result["model"], records)
    weights = weights_from_predictions(pred, as_of, fit_result["policy"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--out", default="stage2_report/experiments/lstm_rank_weight.csv")
    parser.add_argument(
        "--policy-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Trading-day horizon used to select top_k/temperature/rank_blend on validation windows.",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Trading-day horizon for direct future-rank/future-weight training targets.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit_result = fit_lstm(
        prices,
        index_df,
        as_of,
        policy_horizon=args.policy_horizon,
        target_horizon=args.target_horizon,
    )
    submission = generate_submission(fit_result, as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(">> selected policy")
    print(fit_result["policy"])
    print(">> validation policy table")
    print(fit_result["policy_table"].head(12).to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
