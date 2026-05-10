"""
Lightweight Transformer challenger for direct rank/weight learning.

This mirrors lstm_rank_weight.py's targets and portfolio policy search, but
uses a small Transformer encoder over each stock's recent feature sequence.  The
model is intentionally compact because our available stock panel is short in
calendar time; a large Transformer would be very easy to overfit here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lstm_rank_weight as lstm
from features import FORWARD_HORIZON, build_features

DATA_DIR = ROOT / "data"

EPOCHS = 7
BATCH_SIZE = 640
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.18


class TransformerRankWeight(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int = lstm.SEQ_LEN,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, d_model),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.rank_head = nn.Linear(d_model, 1)
        self.weight_head = nn.Linear(d_model, 1)

    def forward(self, seq):
        x = self.input_proj(seq) + self.pos[:, : seq.shape[1], :]
        encoded = self.encoder(x)
        # Mean pooling is more stable than relying on only the final token for
        # short-horizon noisy financial sequences.
        h = self.head(encoded.mean(dim=1))
        return self.rank_head(h).squeeze(-1), self.weight_head(h).squeeze(-1)


def train_model(train_records: list[lstm.Sample], val_records: list[lstm.Sample]):
    lstm.set_seed()
    torch.set_num_threads(4)
    model = TransformerRankWeight(len(lstm.SEQUENCE_FEATURES)).to(lstm.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    train_loader = DataLoader(lstm.SequenceDataset(train_records), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(lstm.SequenceDataset(val_records), batch_size=BATCH_SIZE, shuffle=False)
    best_state = None
    best_val = float("inf")
    wait = 0
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for seq, rank_t, weight_t in train_loader:
            seq = seq.to(lstm.DEVICE)
            rank_t = rank_t.to(lstm.DEVICE)
            weight_t = weight_t.to(lstm.DEVICE)
            rank_pred, weight_pred = model(seq)
            rank_loss = F.smooth_l1_loss(rank_pred, rank_t)
            weight_loss = F.mse_loss(torch.relu(weight_pred), torch.sqrt(weight_t))
            top_loss = (((rank_pred - rank_t) ** 2) * (1.0 + 2.5 * (rank_t > 0.80).float())).mean()
            sample_weight = 1.0 + 5.0 * torch.sqrt(weight_t)
            loss = (rank_loss + 0.65 * weight_loss + 0.18 * top_loss) * sample_weight.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq, rank_t, weight_t in val_loader:
                rank_pred, weight_pred = model(seq.to(lstm.DEVICE))
                rank_t = rank_t.to(lstm.DEVICE)
                weight_t = weight_t.to(lstm.DEVICE)
                rank_loss = F.smooth_l1_loss(rank_pred, rank_t)
                weight_loss = F.mse_loss(torch.relu(weight_pred), torch.sqrt(weight_t))
                val_losses.append(float((rank_loss + 0.65 * weight_loss).item()))
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


def fit_transformer(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp):
    min_date = as_of - pd.Timedelta(days=lstm.LOOKBACK_DAYS)
    prices = prices[prices["date"] >= min_date].copy()
    index_df = index_df[index_df["date"] >= min_date].copy()
    panel = lstm.add_direct_targets(build_features(prices, index_df), index_df)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    usable = panel.dropna(subset=lstm.SEQUENCE_FEATURES + ["target_rank_5d", "target_weight_5d"])
    train_pool = usable[usable["date"] <= train_cutoff].copy()
    train_dates, val_dates, train_end, val_start = lstm.date_splits(train_pool)
    train_records_raw = lstm.build_records(panel, train_dates)
    val_records_raw = lstm.build_records(panel, val_dates)
    normalizer = lstm.Normalizer(train_records_raw)
    train_records = normalizer.apply(train_records_raw)
    val_records = normalizer.apply(val_records_raw)
    print(f">> device={lstm.DEVICE} model=transformer train_records={len(train_records):,} val_records={len(val_records):,}")
    print(f">> train_end={train_end.date()} val_start={val_start.date()}")
    model = train_model(train_records, val_records)
    val_pred = lstm.predict_records(model, val_records)
    policy, policy_table = lstm.select_policy(val_pred, val_dates, trading_dates, prices, index_df)
    return {
        "panel": panel,
        "trading_dates": trading_dates,
        "normalizer": normalizer,
        "model": model,
        "policy": policy,
        "policy_table": policy_table,
        "prices": prices,
        "index_df": index_df,
    }


def generate_submission(fit_result: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    records = lstm.build_records(fit_result["panel"], [as_of])
    records = fit_result["normalizer"].apply(records)
    pred = lstm.predict_records(fit_result["model"], records)
    weights = lstm.weights_from_predictions(pred, as_of, fit_result["policy"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--out", default="submissions/experiments/transformer_rank_weight.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit_result = fit_transformer(prices, index_df, as_of)
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
