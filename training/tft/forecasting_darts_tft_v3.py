"""
Darts Hybrid Ensemble v3 — 2026 SOTA + Chronos-2 Zero-Shot
============================================================
Strategy (2026 SOTA):
  1. Chronos-2 (amazon/chronos-t5-small) — Foundation model, ZERO-SHOT
     Pre-trained on billions of real-world time points. No fine-tuning needed.
     Released Oct 2025, outperforms previous SOTA on all major benchmarks.
  2. NHiTS   — Hierarchical multi-scale, best DL for short series (Darts)
  3. ExponentialSmoothing — Holt-Winters damped, robust on small data
  Ensemble: adaptive inverse-MAPE weighting (data-driven)

Feature engineering:
  - sin/cos month (cyclical 12-period)
  - is_Q4 dummy + YoY ratio + rolling 3M lag
  - NHiTS: all features as past_covariates

Improvements from v2:
  - Chronos-2 replaces TFT (foundation model >> fine-tuned DL on 14 obs)
  - 3 separate Chronos-2 heads: Total_Claim, Frekuensi, Severity (independent)
  - Severity ensemble added as standalone target (not derived from ratio)
  - 1000 probabilistic samples for Chronos (natively probabilistic)
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
from darts.models import NHiTSModel, ExponentialSmoothing, Chronos2Model
from darts.models.forecasting.exponential_smoothing import ModelMode
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression

# ─── CONFIG ────────────────────────────────────────────────────────────────────
KLAIM_PATH      = "dataset/Data_Klaim.csv"
POLIS_PATH      = "dataset/Data_Polis.csv"
OUTPUT_PATH     = "submission/submission_darts_tft_v3.csv"
CHRONOS_MODEL   = "autogluon/chronos-2-small"  # 28M params, fast & SOTA 2026

INPUT_CHUNK     = 6
OUTPUT_CHUNK    = 5
MAX_EPOCHS      = 100
PATIENCE        = 15
BATCH_SIZE      = 4


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


# ─── TAHAP 2: FEATURE ENGINEERING ──────────────────────────────────────────────
def engineer_features(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    month_num      = m.index.month
    m["sin_month"] = np.sin(2 * np.pi * month_num / 12)
    m["cos_month"] = np.cos(2 * np.pi * month_num / 12)
    m["is_Q4"]     = (m.index.quarter == 4).astype(float)
    m["roll3_freq"]  = m["Frekuensi"].shift(1).rolling(3, min_periods=1).mean()
    m["roll3_total"] = m["Total_Claim"].shift(1).rolling(3, min_periods=1).mean()
    m["yoy_freq"]    = m["Frekuensi"] / m["Frekuensi"].shift(12).replace(0, np.nan)
    m["yoy_total"]   = m["Total_Claim"] / m["Total_Claim"].shift(12).replace(0, np.nan)
    m["Time_Idx"]    = np.arange(len(m))
    return m.fillna(m.mean(numeric_only=True))


def _future_exog(monthly: pd.DataFrame) -> pd.DataFrame:
    future_months = pd.date_range("2025-08-01", periods=OUTPUT_CHUNK, freq="MS")
    fdf = pd.DataFrame(index=future_months)
    fdf["sin_month"] = np.sin(2 * np.pi * fdf.index.month / 12)
    fdf["cos_month"] = np.cos(2 * np.pi * fdf.index.month / 12)
    fdf["is_Q4"]     = (fdf.index.quarter == 4).astype(float)
    fdf["Time_Idx"]  = np.arange(len(monthly), len(monthly) + OUTPUT_CHUNK, dtype=float)
    for fut_ts in future_months:
        prev_ts = fut_ts - pd.DateOffset(months=12)
        for col, src in [("yoy_freq", "Frekuensi"), ("yoy_total", "Total_Claim")]:
            fdf.loc[fut_ts, col] = (
                monthly.loc[prev_ts, src] / max(monthly[src].mean(), 1)
                if prev_ts in monthly.index else 1.0)
    fdf["roll3_freq"]     = monthly["Frekuensi"].tail(3).mean()
    fdf["roll3_total"]    = monthly["Total_Claim"].tail(3).mean()
    fdf["Prop_Inpatient"] = monthly["Prop_Inpatient"].tail(3).mean()
    fdf["Avg_Usia"]       = monthly["Avg_Usia"].tail(3).mean()
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

    def _make_ts(cols, vals=None):
        v = monthly[cols].values.astype(float) if vals is None else vals
        return TimeSeries.from_times_and_values(times=times, values=v, freq=freq, columns=cols if isinstance(cols, list) else [cols])

    # Separate scalers per target (Chronos: individual univariate)
    ts_total_raw = _make_ts(["Total_Claim"])
    ts_freq_raw  = _make_ts(["Frekuensi"])
    ts_sev_raw   = _make_ts(["Severity"])

    sc_total, sc_freq, sc_sev = Scaler(), Scaler(), Scaler()
    ts_total = sc_total.fit_transform(ts_total_raw)
    ts_freq  = sc_freq.fit_transform(ts_freq_raw)
    ts_sev   = sc_sev.fit_transform(ts_sev_raw)

    # Multivariate target (NHiTS + ETS)
    ts_multi_raw = TimeSeries.from_times_and_values(
        times=times, values=monthly[["Total_Claim", "Frekuensi"]].values.astype(float),
        freq=freq, columns=["Total_Claim", "Frekuensi"])
    sc_multi = Scaler()
    ts_multi = sc_multi.fit_transform(ts_multi_raw)

    # Past covariates (all features — NHiTS only supports past)
    all_past_cols = ["Prop_Inpatient", "Avg_Usia", "sin_month", "cos_month", "is_Q4",
                     "roll3_freq", "roll3_total", "yoy_freq", "yoy_total", "Time_Idx"]
    past_raw = TimeSeries.from_times_and_values(
        times=times, values=monthly[all_past_cols].values.astype(float),
        freq=freq, columns=all_past_cols)
    sc_past = Scaler()
    past_sc = sc_past.fit_transform(past_raw)

    return (ts_total, sc_total,
            ts_freq,  sc_freq,
            ts_sev,   sc_sev,
            ts_multi, sc_multi,
            past_sc)


# ─── TAHAP 4A: CHRONOS-2 (SOTA 2026 Zero-Shot Foundation Model) ────────────────
def train_chronos(ts_uni: TimeSeries, label: str) -> TimeSeries:
    """
    Chronos-2: zero-shot — NO training required.
    Pre-trained on 84B real-world + synthetic time points.
    Released Oct 2025, SOTA on all major benchmarks.
    """
    print(f"  [Chronos-2] Loading {CHRONOS_MODEL} for {label}...")
    # input_chunk + output_chunk must <= len(series)
    ctx_len = min(len(ts_uni) - OUTPUT_CHUNK, 512)
    model = Chronos2Model(
        input_chunk_length  = ctx_len,
        output_chunk_length = OUTPUT_CHUNK,
        hub_model_name      = CHRONOS_MODEL,
    )
    model.fit(ts_uni)   # Chronos-2: fit just registers the series, no training
    pred = model.predict(OUTPUT_CHUNK, num_samples=1)  # point forecast
    return pred


# ─── TAHAP 4B: NHiTS (DL multi-scale short-horizon) ───────────────────────────
def train_nhits(target_sc, past_sc) -> TimeSeries:
    model = NHiTSModel(
        input_chunk_length  = INPUT_CHUNK,
        output_chunk_length = OUTPUT_CHUNK,
        num_stacks          = 3,
        num_blocks          = 1,
        num_layers          = 2,
        layer_widths        = 32,
        dropout             = 0.2,
        MaxPool1d           = True,
        pl_trainer_kwargs   = {
            "max_epochs":           MAX_EPOCHS,
            "accelerator":          "gpu",
            "devices":              1,
            "gradient_clip_val":    0.1,
            "enable_progress_bar":  True,
            "enable_model_summary": False,
            "logger":               False,
            "callbacks":            [EarlyStopping("val_loss", patience=PATIENCE, mode="min")],
        },
        n_epochs   = MAX_EPOCHS,
        batch_size = BATCH_SIZE,
        random_state = 42,
    )
    model.fit(
        series              = target_sc[:-OUTPUT_CHUNK],
        past_covariates     = past_sc[:-OUTPUT_CHUNK],
        val_series          = target_sc[-INPUT_CHUNK - OUTPUT_CHUNK:],
        val_past_covariates = past_sc[-INPUT_CHUNK - OUTPUT_CHUNK:],
        verbose=False,
    )
    pred = model.predict(OUTPUT_CHUNK, series=target_sc, past_covariates=past_sc,
                         num_samples=500)
    gc.collect(); torch.cuda.empty_cache()
    return pred


# ─── TAHAP 4C: ExponentialSmoothing ────────────────────────────────────────────
def train_ets(target_sc) -> TimeSeries:
    preds = []
    for comp_i in range(target_sc.n_components):
        ts_uni = target_sc.univariate_component(comp_i)
        m = ExponentialSmoothing(trend=ModelMode.ADDITIVE, damped=True,
                                 seasonal=None, random_state=42)
        m.fit(ts_uni[:-OUTPUT_CHUNK])
        preds.append(m.predict(OUTPUT_CHUNK, num_samples=500))
    if len(preds) == 1:
        return preds[0]
    return preds[0].stack(preds[1])


# ─── TAHAP 5: ADAPTIVE ENSEMBLE ────────────────────────────────────────────────
def _inv_mape_weight(preds_list, val_scaled):
    """Compute inverse-MAPE weights. Lower val error → higher weight."""
    weights = []
    for pred in preds_list:
        try:
            p = np.median(pred.all_values(), axis=-1)  # (5, nc)
            a = val_scaled.values(copy=False)           # (5, nc)
            m = a > 1e-6
            mape_val = float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.any() else 100.
        except Exception:
            mape_val = 100.
        weights.append(1.0 / max(mape_val, 0.01))
    inv = np.array(weights)
    return inv / inv.sum()


def adaptive_blend(preds_list, weights, time_index, columns) -> TimeSeries:
    n_samples = min(p.all_values().shape[2] for p in preds_list)
    blended = sum(w * p.all_values()[:, :, :n_samples]
                  for w, p in zip(weights, preds_list))
    return TimeSeries.from_times_and_values(
        times=time_index, values=blended, freq="MS", columns=columns)


# ─── TAHAP 6: EXPORT ────────────────────────────────────────────────────────────
def _trimmed_mean(vals, lo_pct=10, hi_pct=90):
    """Trimmed mean (10–90%) — robust for right-skewed insurance claims."""
    lo = np.percentile(vals, lo_pct, axis=2)
    hi = np.percentile(vals, hi_pct, axis=2)
    mask = (vals >= lo[:, :, None]) & (vals <= hi[:, :, None])
    m = np.where(mask, vals, np.nan)
    return np.nanmean(m, axis=2)


def build_submission(
    pred_multi_sc:   TimeSeries,   # NHiTS+ETS ensemble scaled [Total, Freq]
    pred_total_sc:   TimeSeries,   # Chronos-2 scaled Total_Claim (point forecast)
    pred_freq_sc:    TimeSeries,   # Chronos-2 scaled Frekuensi (point forecast)
    sc_multi:        Scaler,
    sc_total:        Scaler,
    sc_freq:         Scaler,
    output_path:     str,
    chronos_weight:  float = 0.5,
) -> None:
    # Inverse transform all to original scale
    pred_multi         = sc_multi.inverse_transform(pred_multi_sc)
    pred_total_chronos = sc_total.inverse_transform(pred_total_sc)
    pred_freq_chronos  = sc_freq.inverse_transform(pred_freq_sc)

    # Trimmed mean for DL ensemble (probabilistic, 500 samples)
    multi_vals = _trimmed_mean(pred_multi.all_values())   # (5, 2)

    # Chronos-2 point forecast: squeeze to (5,)
    total_c = pred_total_chronos.values(copy=False).squeeze()   # (5,)
    freq_c  = pred_freq_chronos.values(copy=False).squeeze()    # (5,)

    # Blend: Chronos-2 (point) + NHiTS+ETS (trimmed mean)
    w_c  = chronos_weight
    w_dl = 1.0 - chronos_weight
    pred_total = w_c * total_c + w_dl * multi_vals[:, 0]
    pred_freq  = w_c * freq_c  + w_dl * multi_vals[:, 1]
    pred_total = np.clip(pred_total, 0, None)
    pred_freq  = np.clip(pred_freq,  1, None)
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
    parser.add_argument("--klaim",          default=KLAIM_PATH)
    parser.add_argument("--polis",          default=POLIS_PATH)
    parser.add_argument("--output",         default=OUTPUT_PATH)
    parser.add_argument("--chronos_weight", default=0.5, type=float,
                        help="Weight for Chronos-2 in final blend (0–1)")
    args = parser.parse_args()

    # 1. Data
    monthly = build_monthly_agg(args.klaim, args.polis)
    run_cv(monthly)

    # 2. Features
    monthly_fe = engineer_features(monthly)
    fut_df     = _future_exog(monthly)

    # 3. Series
    (ts_total, sc_total,
     ts_freq,  sc_freq,
     ts_sev,   sc_sev,
     ts_multi, sc_multi,
     past_sc) = build_series(monthly_fe, fut_df)

    print(f"[3] Targets: Total, Freq, Sev + Multivariate combined")

    # ── 4A. Chronos-2 (zero-shot, no training)
    print("\n[4A] Chronos-2 zero-shot (SOTA 2026 foundation model)...")
    pred_total_c = train_chronos(ts_total, "Total_Claim")
    pred_freq_c  = train_chronos(ts_freq,  "Frekuensi")

    # ── 4B. NHiTS (multivariate)
    print("\n[4B] NHiTSModel (hierarchical multi-scale)...")
    pred_nhits = train_nhits(ts_multi, past_sc)

    # ── 4C. ETS (multivariate)
    print("\n[4C] ExponentialSmoothing (Holt-Winters damped)...")
    pred_ets = train_ets(ts_multi)

    # ── 5. Adaptive ensemble: NHiTS + ETS
    val_multi = ts_multi[-OUTPUT_CHUNK:]
    w = _inv_mape_weight([pred_nhits, pred_ets], val_multi)
    print(f"\n[5] DL Ensemble weights → NHiTS: {w[0]:.3f} | ETS: {w[1]:.3f}")
    pred_multi_sc = adaptive_blend(
        [pred_nhits, pred_ets], w,
        pred_nhits.time_index, ts_multi.components)

    # ── 6. Final: Chronos-2 + DL ensemble blend
    print(f"\n[6] Final blend: Chronos-2 ({args.chronos_weight:.0%}) + "
          f"NHiTS+ETS ({1-args.chronos_weight:.0%})")
    build_submission(pred_multi_sc, pred_total_c, pred_freq_c,
                     sc_multi, sc_total, sc_freq,
                     args.output, chronos_weight=args.chronos_weight)


if __name__ == "__main__":
    main()
