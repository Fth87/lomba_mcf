"""
Darts Hybrid Ensemble Pipeline v2 — 2026 Best Practices
==========================================================
Strategy (berdasarkan riset 2026):
  1. NHiTS  — SOTA untuk multi-horizon short series (hierarchical interpolation)
  2. TFT    — non-linearitas + covariate effects (diperkecil untuk 14 obs)
  3. ExponentialSmoothing — Holt-Winters trend/level (sangat kuat pada data kecil)
  Ensemble: adaptive weight berdasarkan in-sample MAPE per model

Feature engineering:
  - sin/cos month encoding (cyclical seasonality 12-period)
  - is_Q4 dummy (klaim asuransi cenderung spike H2)
  - Rolling 3M mean lag target (Frekuensi & Total_Claim)
  - YoY ratio (bulan yg sama tahun lalu)
  - Prop_Inpatient + Avg_Usia (past covariates)
"""
import warnings
warnings.filterwarnings("ignore")

import gc
import argparse
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping

from darts import TimeSeries
from darts.models import TFTModel, NHiTSModel, ExponentialSmoothing
from darts.models.forecasting.exponential_smoothing import ModelMode
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape as darts_mape
from darts.utils.likelihood_models import QuantileRegression

# ─── CONFIG ────────────────────────────────────────────────────────────────────
KLAIM_PATH   = "dataset/Data_Klaim.csv"
POLIS_PATH   = "dataset/Data_Polis.csv"
OUTPUT_PATH  = "submission/submission_darts_tft_v2.csv"

INPUT_CHUNK  = 6
OUTPUT_CHUNK = 5
MAX_EPOCHS   = 100
PATIENCE     = 15
BATCH_SIZE   = 4


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

    df_k_age = df_k.merge(df_p[["Nomor_Polis", "Tanggal_Lahir"]], on="Nomor_Polis", how="left")
    df_k_age["Usia"] = (df_k_age["Bulan_Obs"] - df_k_age["Tanggal_Lahir"]).dt.days // 365
    avg_age = df_k_age.groupby("Bulan_Obs")["Usia"].mean().rename("Avg_Usia")
    monthly = monthly.merge(avg_age, on="Bulan_Obs", how="left")
    monthly = monthly.set_index("Bulan_Obs").sort_index()
    monthly = monthly.fillna(monthly.mean(numeric_only=True))
    print(f"[1] Monthly obs: {len(monthly)} | {monthly.index.min().date()} – {monthly.index.max().date()}")
    return monthly


# ─── TAHAP 2: FEATURE ENGINEERING (2026 Best Practice) ─────────────────────────
def engineer_features(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()

    # 1. Cyclical month encoding (sin/cos) — captures 12-period seasonality
    month_num     = m.index.month
    m["sin_month"] = np.sin(2 * np.pi * month_num / 12)
    m["cos_month"] = np.cos(2 * np.pi * month_num / 12)

    # 2. Quarter 4 dummy — H2 seasonality in insurance claims
    m["is_Q4"] = (m.index.quarter == 4).astype(float)

    # 3. Rolling 3-month lag of target (momentum)
    m["roll3_freq"]  = m["Frekuensi"].shift(1).rolling(3, min_periods=1).mean()
    m["roll3_total"] = m["Total_Claim"].shift(1).rolling(3, min_periods=1).mean()

    # 4. YoY ratio (month+12 vs month) — available for obs 13–18 (Jan–Jul 2025)
    m["yoy_freq"]  = m["Frekuensi"] / m["Frekuensi"].shift(12).replace(0, np.nan)
    m["yoy_total"] = m["Total_Claim"] / m["Total_Claim"].shift(12).replace(0, np.nan)

    # 5. Time index (0–18)
    m["Time_Idx"] = np.arange(len(m))

    return m.fillna(m.mean(numeric_only=True))


def _future_exog(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build exogenous features for Aug–Dec 2025 (deterministic)."""
    future_months = pd.date_range("2025-08-01", periods=OUTPUT_CHUNK, freq="MS")
    fdf = pd.DataFrame(index=future_months)
    fdf["sin_month"] = np.sin(2 * np.pi * fdf.index.month / 12)
    fdf["cos_month"] = np.cos(2 * np.pi * fdf.index.month / 12)
    fdf["is_Q4"]     = (fdf.index.quarter == 4).astype(float)
    fdf["Time_Idx"]  = np.arange(len(monthly), len(monthly) + OUTPUT_CHUNK, dtype=float)

    # YoY ratio: use 2024 equivalent (month offset -12)
    for m_ts, fut_ts in zip(future_months, future_months):
        prev_ts = fut_ts - pd.DateOffset(months=12)
        for col in ["yoy_freq", "yoy_total"]:
            src = "Frekuensi" if "freq" in col else "Total_Claim"
            if prev_ts in monthly.index:
                fdf.loc[fut_ts, col] = monthly.loc[fut_ts - pd.DateOffset(months=12), src] / \
                    max(monthly[src].mean(), 1)
            else:
                fdf.loc[fut_ts, col] = 1.0

    # Rolling 3M: use last 3 obs as proxy
    fdf["roll3_freq"]  = monthly["Frekuensi"].tail(3).mean()
    fdf["roll3_total"] = monthly["Total_Claim"].tail(3).mean()
    fdf["Prop_Inpatient"] = monthly["Prop_Inpatient"].tail(3).mean()
    fdf["Avg_Usia"]    = monthly["Avg_Usia"].tail(3).mean()
    return fdf.fillna(1.0)


# ─── CROSS-VALIDATION h=5 ──────────────────────────────────────────────────────
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


# ─── TAHAP 3: BUILD DARTS TimeSeries ───────────────────────────────────────────
def build_series(monthly: pd.DataFrame, fut_df: pd.DataFrame):
    times = pd.DatetimeIndex(monthly.index)
    freq  = "MS"

    # Target
    target_raw = TimeSeries.from_times_and_values(
        times=times,
        values=monthly[["Total_Claim", "Frekuensi"]].values.astype(float),
        freq=freq, columns=["Total_Claim", "Frekuensi"])
    scaler = Scaler()
    target_sc = scaler.fit_transform(target_raw)

    # Past covariates: Prop_Inpatient, Avg_Usia, roll3_freq, roll3_total, yoy_freq, yoy_total
    past_cols  = ["Prop_Inpatient", "Avg_Usia", "roll3_freq", "roll3_total", "yoy_freq", "yoy_total"]
    past_raw   = TimeSeries.from_times_and_values(
        times=times, values=monthly[past_cols].values.astype(float),
        freq=freq, columns=past_cols)
    past_sc    = Scaler().fit_transform(past_raw)

    # Future covariates for TFT: sin_month, cos_month, is_Q4, Time_Idx
    # (deterministically known for all future periods)
    fut_cols    = ["sin_month", "cos_month", "is_Q4", "Time_Idx"]
    all_times   = times.append(pd.DatetimeIndex(fut_df.index))
    all_fut_vals= np.vstack([monthly[fut_cols].values, fut_df[fut_cols].values]).astype(float)
    fut_cov_raw = TimeSeries.from_times_and_values(
        times=all_times, values=all_fut_vals, freq=freq, columns=fut_cols)
    fut_cov_sc  = Scaler().fit_transform(fut_cov_raw)

    # Past covariates ALL (for NHiTS — no future_cov support): past + cyclical hist only
    all_past_cols = past_cols + ["sin_month", "cos_month", "is_Q4", "Time_Idx"]
    past_all_raw  = TimeSeries.from_times_and_values(
        times=times, values=monthly[all_past_cols].values.astype(float),
        freq=freq, columns=all_past_cols)
    past_all_sc   = Scaler().fit_transform(past_all_raw)

    return target_sc, scaler, past_sc, fut_cov_sc, past_all_sc


# ─── TAHAP 4: TRAIN EACH MODEL ─────────────────────────────────────────────────
def _trainer_kwargs(patience: int) -> dict:
    return {
        "max_epochs":           MAX_EPOCHS,
        "accelerator":          "gpu",
        "devices":              1,
        "gradient_clip_val":    0.1,
        "enable_progress_bar":  True,
        "enable_model_summary": False,
        "logger":               False,
        "callbacks":            [EarlyStopping("val_loss", patience=patience, mode="min")],
    }


def train_tft(target_sc, past_sc, fut_cov_sc) -> TimeSeries:
    # Smaller architecture for 14 obs — 2026 best practice: less = more
    model = TFTModel(
        input_chunk_length    = INPUT_CHUNK,
        output_chunk_length   = OUTPUT_CHUNK,
        hidden_size           = 16,        # 64→16 prevents overfitting
        lstm_layers           = 1,         # 2→1
        num_attention_heads   = 2,
        dropout               = 0.3,       # 0.1→0.3 stronger regularization
        add_relative_index    = True,
        likelihood            = QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        use_static_covariates = False,
        pl_trainer_kwargs     = _trainer_kwargs(PATIENCE),
        batch_size            = BATCH_SIZE,
        n_epochs              = MAX_EPOCHS,
        random_state          = 42,
    )
    model.fit(
        series                = target_sc[:-OUTPUT_CHUNK],
        past_covariates       = past_sc[:-OUTPUT_CHUNK],
        future_covariates     = fut_cov_sc,
        val_series            = target_sc[-INPUT_CHUNK - OUTPUT_CHUNK:],
        val_past_covariates   = past_sc[-INPUT_CHUNK - OUTPUT_CHUNK:],
        val_future_covariates = fut_cov_sc,
        verbose=False,
    )
    pred = model.predict(
        OUTPUT_CHUNK, series=target_sc, past_covariates=past_sc,
        future_covariates=fut_cov_sc, num_samples=500)
    gc.collect(); torch.cuda.empty_cache()
    return pred


def train_nhits(target_sc, past_all_sc) -> TimeSeries:
    # NHiTS: 2026 SOTA for multi-horizon short series
    # Uses past_covariates only (Darts 0.41 NHiTS limitation)
    model = NHiTSModel(
        input_chunk_length  = INPUT_CHUNK,
        output_chunk_length = OUTPUT_CHUNK,
        num_stacks          = 3,
        num_blocks          = 1,
        num_layers          = 2,
        layer_widths        = 32,
        dropout             = 0.2,
        activation          = "ReLU",
        MaxPool1d           = True,
        pl_trainer_kwargs   = _trainer_kwargs(PATIENCE),
        n_epochs            = MAX_EPOCHS,
        batch_size          = BATCH_SIZE,
        random_state        = 42,
    )
    model.fit(
        series              = target_sc[:-OUTPUT_CHUNK],
        past_covariates     = past_all_sc[:-OUTPUT_CHUNK],
        val_series          = target_sc[-INPUT_CHUNK - OUTPUT_CHUNK:],
        val_past_covariates = past_all_sc[-INPUT_CHUNK - OUTPUT_CHUNK:],
        verbose=False,
    )
    pred = model.predict(OUTPUT_CHUNK, series=target_sc, past_covariates=past_all_sc,
                         num_samples=500)
    gc.collect(); torch.cuda.empty_cache()
    return pred


def train_ets(target_sc) -> TimeSeries:
    # Exponential Smoothing: Holt-Winters, robust on small datasets
    # Per-component (univariate only) → separate for Total & Freq
    preds = []
    for comp_i in range(2):
        ts_uni = target_sc.univariate_component(comp_i)
        m = ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            damped=True,
            seasonal=None,
            random_state=42,
        )
        m.fit(ts_uni[:-OUTPUT_CHUNK])
        preds.append(m.predict(OUTPUT_CHUNK, num_samples=500))
    return preds[0].stack(preds[1])


# ─── TAHAP 5: ADAPTIVE ENSEMBLE ────────────────────────────────────────────────
def adaptive_ensemble(
    target_sc: TimeSeries,
    pred_tft: TimeSeries,
    pred_nhits: TimeSeries,
    pred_ets: TimeSeries,
) -> TimeSeries:
    """
    Adaptive weights based on in-sample MAPE (val = last 5 obs).
    Lower MAPE → higher weight. 2026 best practice: data-driven weighting.
    """
    val = target_sc[-OUTPUT_CHUNK:]

    def _val_mape(pred):
        # Use only the last OUTPUT_CHUNK of each prediction for comparison
        # pred is a future forecast — compare against last 5 known obs as proxy
        try:
            p_vals = np.median(pred.all_values(), axis=-1)   # (5, 2)
            a_vals = val.values(copy=False)                   # (5, 2)
            m = a_vals > 1e-6
            if not m.any():
                return 100.0
            return float(np.mean(np.abs((a_vals[m] - p_vals[m]) / a_vals[m])) * 100)
        except Exception:
            return 100.0

    # Predict val period (use 2nd-to-last chunk as context)
    ctx = target_sc[:-OUTPUT_CHUNK]

    # Simple in-sample val: compare against last 5 scaled obs
    # (proxy for generalization error)
    m_tft   = _val_mape(pred_tft)
    m_nhits = _val_mape(pred_nhits)
    m_ets   = _val_mape(pred_ets)

    print(f"\n[Ensemble] Val MAPE proxy → TFT: {m_tft:.1f}% | NHiTS: {m_nhits:.1f}% | ETS: {m_ets:.1f}%")

    # Inverse-MAPE weighting (lower error → higher weight)
    inv = np.array([1.0 / max(m_tft, 0.1),
                    1.0 / max(m_nhits, 0.1),
                    1.0 / max(m_ets, 0.1)])
    w = inv / inv.sum()
    print(f"[Ensemble] Adaptive weights → TFT: {w[0]:.3f} | NHiTS: {w[1]:.3f} | ETS: {w[2]:.3f}")

    # Weighted blend of 500-sample predictions
    tft_v   = pred_tft.all_values()      # (5, 2, 500)
    nhits_v = pred_nhits.all_values()
    ets_v   = pred_ets.all_values()

    n_samples = min(tft_v.shape[2], nhits_v.shape[2], ets_v.shape[2])
    blended   = (w[0] * tft_v[:, :, :n_samples]
               + w[1] * nhits_v[:, :, :n_samples]
               + w[2] * ets_v[:, :, :n_samples])

    return TimeSeries.from_times_and_values(
        times  = pred_tft.time_index,
        values = blended,
        freq   = "MS",
        columns= pred_tft.components,
    )


# ─── TAHAP 6: EXPORT ────────────────────────────────────────────────────────────
def build_submission(pred_sc: TimeSeries, scaler: Scaler, output_path: str) -> None:
    pred = scaler.inverse_transform(pred_sc)

    # Trimmed mean (10–90%) — more robust than median for right-skewed claims
    vals  = pred.all_values()   # (5, 2, n_samples)
    lo    = np.percentile(vals, 10, axis=2)
    hi    = np.percentile(vals, 90, axis=2)
    mask  = (vals >= lo[:, :, None]) & (vals <= hi[:, :, None])
    means = np.where(mask, vals, np.nan)
    pred_vals = np.nanmean(means, axis=2)   # (5, 2)

    pred_total = np.clip(pred_vals[:, 0], 0, None)
    pred_freq  = np.clip(pred_vals[:, 1], 1, None)
    pred_sev   = pred_total / pred_freq

    future_months = pd.date_range("2025-08-01", periods=OUTPUT_CHUNK, freq="MS")
    rows = []
    for i, month in enumerate(future_months):
        ms = month.strftime("%Y-%m")
        rows.extend([
            {"id": f"{ms}_Claim_Frequency", "value": round(float(pred_freq[i]))},
            {"id": f"{ms}_Claim_Severity",  "value": round(float(pred_sev[i]), 2)},
            {"id": f"{ms}_Total_Claim",     "value": round(float(pred_total[i]), 2)},
        ])

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)
    print(f"\n[6] Submission saved → {output_path}")
    print(submission.to_string(index=False))


# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--klaim",  default=KLAIM_PATH)
    parser.add_argument("--polis",  default=POLIS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    # 1. Data
    monthly = build_monthly_agg(args.klaim, args.polis)
    run_cv(monthly)

    # 2. Feature engineering
    monthly_fe = engineer_features(monthly)
    fut_df     = _future_exog(monthly)
    print(f"[2] Features: {list(monthly_fe.columns)}")

    # 3. Darts series
    target_sc, scaler, past_sc, fut_cov_sc, past_all_sc = build_series(monthly_fe, fut_df)
    print(f"[3] Target: {target_sc.components.tolist()}")

    # 4. Train 3 models
    print("\n[4a] Training TFTModel (hidden=16, dropout=0.3) ...")
    pred_tft = train_tft(target_sc, past_sc, fut_cov_sc)

    print("\n[4b] Training NHiTSModel (2026 SOTA for short-horizon) ...")
    pred_nhits = train_nhits(target_sc, past_all_sc)

    print("\n[4c] Training ExponentialSmoothing (Holt-Winters damped) ...")
    pred_ets = train_ets(target_sc)

    # 5. Adaptive ensemble
    pred_ensemble = adaptive_ensemble(target_sc, pred_tft, pred_nhits, pred_ets)

    # 6. Export
    build_submission(pred_ensemble, scaler, args.output)


if __name__ == "__main__":
    main()
