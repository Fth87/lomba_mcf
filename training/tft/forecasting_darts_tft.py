"""
Darts TFT Pipeline — Multi-Horizon Insurance Claim Forecasting
Strategy: aggregate to monthly portfolio level (19 obs), then use TFTModel
with multivariate target [Total_Claim, Frekuensi] + time-varying covariates.
Normalization handled by Darts built-in Scaler (no log1p on billion-scale targets).
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression

# ─── CONFIG ────────────────────────────────────────────────────────────────────
KLAIM_PATH   = "dataset/Data_Klaim.csv"
POLIS_PATH   = "dataset/Data_Polis.csv"
OUTPUT_PATH  = "submission/submission_darts_tft.csv"

INPUT_CHUNK  = 6    # encoder: look back 6 months
OUTPUT_CHUNK = 5    # decoder: forecast Aug–Dec 2025
MAX_EPOCHS   = 80
PATIENCE     = 12
BATCH_SIZE   = 4    # only 14 training obs → small batch


# ─── TAHAP 1: MONTHLY PORTFOLIO AGGREGATION + p98 CAP ──────────────────────────
def build_monthly_agg(klaim_path: str, polis_path: str) -> pd.DataFrame:
    df_k = pd.read_csv(klaim_path)
    df_p = pd.read_csv(polis_path)

    df_k["Tanggal Pasien Masuk RS"] = pd.to_datetime(
        df_k["Tanggal Pasien Masuk RS"], errors="coerce")
    df_p["Tanggal_Lahir"] = pd.to_datetime(
        df_p["Tanggal Lahir"].astype(str), format="%Y%m%d", errors="coerce")
    df_p = df_p.rename(columns={"Nomor Polis": "Nomor_Polis", "Plan Code": "Plan_Code"})
    df_k = df_k.rename(columns={"Nomor Polis": "Nomor_Polis"})

    # p98 cap at individual level BEFORE aggregation
    p98 = df_k["Nominal Klaim Yang Disetujui"].quantile(0.98)
    df_k["Nominal_Final"] = df_k["Nominal Klaim Yang Disetujui"].clip(upper=p98)
    print(f"[1] p98 cap = {p98:,.0f}")

    df_k["is_inpatient"] = (df_k["Inpatient/Outpatient"].str.strip().str.upper() == "IP").astype(float)
    df_k["Bulan_Obs"]    = df_k["Tanggal Pasien Masuk RS"].dt.to_period("M").dt.to_timestamp()

    monthly = df_k.groupby("Bulan_Obs").agg(
        Total_Claim    = ("Nominal_Final",  "sum"),
        Frekuensi      = ("Nominal_Final",  "count"),
        sum_ip         = ("is_inpatient",   "sum"),
        cnt_ip         = ("is_inpatient",   "count"),
    ).reset_index()
    monthly["Severity"]       = monthly["Total_Claim"] / monthly["Frekuensi"].replace(0, np.nan)
    monthly["Prop_Inpatient"] = monthly["sum_ip"] / monthly["cnt_ip"].replace(0, np.nan)

    # Average polis age per month
    df_k_age = df_k.merge(df_p[["Nomor_Polis", "Tanggal_Lahir"]], on="Nomor_Polis", how="left")
    df_k_age["Usia"] = (df_k_age["Bulan_Obs"] - df_k_age["Tanggal_Lahir"]).dt.days // 365
    avg_age = df_k_age.groupby("Bulan_Obs")["Usia"].mean().rename("Avg_Usia")
    monthly = monthly.merge(avg_age, on="Bulan_Obs", how="left")

    monthly["Time_Idx"] = np.arange(len(monthly))
    monthly = monthly.set_index("Bulan_Obs").sort_index()
    monthly = monthly.fillna(monthly.mean(numeric_only=True))
    print(f"[1] Monthly obs: {len(monthly)} | {monthly.index.min().date()} – {monthly.index.max().date()}")
    return monthly


# ─── CROSS-VALIDATION h=5 (baseline mean-6m) ───────────────────────────────────
def run_cv(monthly: pd.DataFrame) -> None:
    def _mape(a, p):
        a, p = np.array(a, float), np.array(p, float)
        m = a > 0
        return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.any() else 0.0

    print("\n[CV] h=5 rolling backtesting (6-month mean baseline)")
    scores = []
    for cutoff in [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]:
        tr = monthly[monthly.index < cutoff]
        te = monthly[monthly.index >= cutoff].head(5)
        if len(te) < 5:
            continue
        mu_f = tr["Frekuensi"].tail(6).mean()
        mu_t = tr["Total_Claim"].tail(6).mean()
        mu_s = mu_t / max(mu_f, 1)
        sf = _mape(te["Frekuensi"],  [mu_f]*5)
        st = _mape(te["Total_Claim"], [mu_t]*5)
        ss = _mape(te["Severity"].fillna(0), [mu_s]*5)
        avg = (sf + st + ss) / 3
        print(f"  cut={cutoff.date()} | Freq={sf:.1f}% Total={st:.1f}% Sev={ss:.1f}% → avg={avg:.1f}%")
        scores.append(avg)
    if scores:
        print(f"[CV] Baseline MAPE: {np.mean(scores):.2f}%\n")


# ─── TAHAP 3: BUILD + SCALE Darts TimeSeries ───────────────────────────────────
def build_darts_series(monthly: pd.DataFrame):
    times = pd.DatetimeIndex(monthly.index)
    freq  = "MS"

    # Target: multivariate [Total_Claim, Frekuensi]
    target_raw = TimeSeries.from_times_and_values(
        times=times,
        values=monthly[["Total_Claim", "Frekuensi"]].values.astype(float),
        freq=freq, columns=["Total_Claim", "Frekuensi"])

    # Scale to [0,1] — Darts Scaler handles inverse transform automatically
    scaler = Scaler()
    target_scaled = scaler.fit_transform(target_raw)

    # Past covariates: [Prop_Inpatient, Avg_Usia] — historically known only
    past_cov_raw = TimeSeries.from_times_and_values(
        times=times,
        values=monthly[["Prop_Inpatient", "Avg_Usia"]].values.astype(float),
        freq=freq, columns=["Prop_Inpatient", "Avg_Usia"])
    past_scaler = Scaler()
    past_cov_scaled = past_scaler.fit_transform(past_cov_raw)

    # Future covariates: Time_Idx — deterministically known for all future months
    all_times  = times.append(pd.date_range("2025-08-01", periods=OUTPUT_CHUNK, freq="MS"))
    fut_vals   = np.arange(len(all_times), dtype=float).reshape(-1, 1)
    future_cov = TimeSeries.from_times_and_values(
        times=all_times, values=fut_vals, freq=freq, columns=["Time_Idx"])
    # Scale future cov consistently
    fc_scaler   = Scaler()
    future_cov_scaled = fc_scaler.fit_transform(future_cov)

    return target_scaled, scaler, past_cov_scaled, future_cov_scaled


# ─── TAHAP 4: TRAIN + INFER ─────────────────────────────────────────────────────
def train_and_predict(
    target_scaled: TimeSeries,
    past_cov_scaled: TimeSeries,
    future_cov_scaled: TimeSeries,
) -> TimeSeries:
    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, mode="min", verbose=False)

    model = TFTModel(
        input_chunk_length    = INPUT_CHUNK,
        output_chunk_length   = OUTPUT_CHUNK,
        hidden_size           = 64,
        lstm_layers           = 2,
        num_attention_heads   = 4,
        dropout               = 0.1,
        add_relative_index    = True,
        likelihood            = QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        use_static_covariates = False,
        pl_trainer_kwargs     = {
            "max_epochs":           MAX_EPOCHS,
            "accelerator":          "gpu",
            "devices":              1,
            "gradient_clip_val":    0.1,
            "enable_progress_bar":  True,
            "enable_model_summary": False,
            "logger":               False,
            "callbacks":            [early_stop],
        },
        batch_size            = BATCH_SIZE,
        n_epochs              = MAX_EPOCHS,
        random_state          = 42,
    )

    train_series   = target_scaled[:-OUTPUT_CHUNK]
    train_past     = past_cov_scaled[:-OUTPUT_CHUNK]

    print(f"[4] Training TFTModel: {len(train_series)} train obs, {OUTPUT_CHUNK} val obs")
    model.fit(
        series                = train_series,
        past_covariates       = train_past,
        future_covariates     = future_cov_scaled,
        val_series            = target_scaled[-INPUT_CHUNK - OUTPUT_CHUNK:],
        val_past_covariates   = past_cov_scaled[-INPUT_CHUNK - OUTPUT_CHUNK:],
        val_future_covariates = future_cov_scaled,
        verbose               = False,
    )

    pred_scaled = model.predict(
        n                 = OUTPUT_CHUNK,
        series            = target_scaled,
        past_covariates   = past_cov_scaled,
        future_covariates = future_cov_scaled,
        num_samples       = 200,
    )
    return pred_scaled


# ─── TAHAP 5: INVERSE SCALE + EXPORT ───────────────────────────────────────────
def build_submission(
    pred_scaled: TimeSeries,
    scaler: Scaler,
    output_path: str,
) -> None:
    pred = scaler.inverse_transform(pred_scaled)

    # Median (q=0.5) across 200 samples → (5 steps, 2 components)
    pred_vals  = np.median(pred.all_values(), axis=-1)
    pred_total = np.clip(pred_vals[:, 0], 0, None)
    pred_freq  = np.clip(pred_vals[:, 1], 1, None)
    pred_sev   = pred_total / pred_freq

    future_months = pd.date_range("2025-08-01", periods=OUTPUT_CHUNK, freq="MS")
    rows = []
    for i, month in enumerate(future_months):
        ms = month.strftime("%Y-%m")   # "2025-08" (DASH format)
        rows.extend([
            {"id": f"{ms}_Claim_Frequency", "value": round(float(pred_freq[i]))},
            {"id": f"{ms}_Claim_Severity",  "value": round(float(pred_sev[i]), 2)},
            {"id": f"{ms}_Total_Claim",     "value": round(float(pred_total[i]), 2)},
        ])

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)
    print(f"\n[5] Submission saved → {output_path}")
    print(submission.to_string(index=False))


# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--klaim",  default=KLAIM_PATH)
    parser.add_argument("--polis",  default=POLIS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    monthly = build_monthly_agg(args.klaim, args.polis)
    run_cv(monthly)

    target_scaled, scaler, past_cov_scaled, future_cov_scaled = build_darts_series(monthly)
    print(f"[3] Target components: {target_scaled.components.tolist()}")

    pred_scaled = train_and_predict(target_scaled, past_cov_scaled, future_cov_scaled)
    build_submission(pred_scaled, scaler, args.output)


if __name__ == "__main__":
    main()
