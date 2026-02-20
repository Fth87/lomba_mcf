"""
Darts Ultra-Ensemble v4 — Trinity Paths & Global Reconciliation
==================================================================
Strategy (2026 SOTA):
  1. Trinity Paths: Memprediksi Frequency (F), Severity (S), dan Total_Claim (T) 
     secara independen (3 ensemble terpisah).
  2. Model Mix: Chronos-2 (SOTA Foundation) + NHiTS + Theta/FourTheta + ETS.
  3. Global Reconciliation: Menyeimbangkan T, F, dan S menggunakan relasi T = F * S.
  4. Momentum Weighting: Menggunakan error 3 bulan terakhir untuk bobot ensemble.

Improvements from v3:
  - Penambahan Theta & FourTheta (Sangat kuat untuk data kecil).
  - Standalone Severity path (menurunkan noise yang biasanya terbawa di Total).
  - Structural consistency via multi-stage reconciliation.
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
from darts.models import NHiTSModel, ExponentialSmoothing, Chronos2Model, Theta, FourTheta
from darts.models.forecasting.exponential_smoothing import ModelMode
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression

# ─── CONFIG ────────────────────────────────────────────────────────────────────
KLAIM_PATH      = "dataset/Data_Klaim.csv"
POLIS_PATH      = "dataset/Data_Polis.csv"
OUTPUT_PATH     = "submission/submission_darts_tft_v4.csv"
CHRONOS_MODEL   = "autogluon/chronos-2-small"

INPUT_CHUNK     = 6
OUTPUT_CHUNK    = 5
MAX_EPOCHS      = 100
PATIENCE        = 15
BATCH_SIZE      = 4


# ─── TAHAP 1: DATA PREP ────────────────────────────────────────────────────────
def build_monthly_agg(klaim_path: str, polis_path: str) -> pd.DataFrame:
    df_k = pd.read_csv(klaim_path)
    df_p = pd.read_csv(polis_path)
    df_k["Tanggal Pasien Masuk RS"] = pd.to_datetime(df_k["Tanggal Pasien Masuk RS"], errors="coerce")
    df_p["Tanggal_Lahir"] = pd.to_datetime(df_p["Tanggal Lahir"].astype(str), format="%Y%m%d", errors="coerce")
    df_k = df_k.rename(columns={"Nomor Polis": "Nomor_Polis"})

    # p98 cap
    p98 = df_k["Nominal Klaim Yang Disetujui"].quantile(0.98)
    df_k["Nominal_Final"] = df_k["Nominal Klaim Yang Disetujui"].clip(upper=p98)

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

    df_k_age = df_k.merge(df_p[["Nomor Polis", "Tanggal_Lahir"]].rename(columns={"Nomor Polis": "Nomor_Polis"}), on="Nomor_Polis", how="left")
    df_k_age["Usia"] = (df_k_age["Bulan_Obs"] - df_k_age["Tanggal_Lahir"]).dt.days // 365
    avg_age = df_k_age.groupby("Bulan_Obs")["Usia"].mean().rename("Avg_Usia")
    monthly = monthly.merge(avg_age, on="Bulan_Obs", how="left")
    monthly = monthly.set_index("Bulan_Obs").sort_index()
    monthly = monthly.fillna(monthly.mean(numeric_only=True))
    return monthly


def engineer_features(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    m["sin_month"] = np.sin(2 * np.pi * m.index.month / 12)
    m["cos_month"] = np.cos(2 * np.pi * m.index.month / 12)
    m["is_Q4"]     = (m.index.quarter == 4).astype(float)
    m["Time_Idx"]    = np.arange(len(m))
    # Rolling & YoY
    m["roll3_f"] = m["Frekuensi"].shift(1).rolling(3, min_periods=1).mean()
    m["roll3_t"] = m["Total_Claim"].shift(1).rolling(3, min_periods=1).mean()
    m["yoy_f"]   = m["Frekuensi"] / m["Frekuensi"].shift(12).replace(0, np.nan)
    return m.fillna(m.mean(numeric_only=True))


# ─── TAHAP 2: MODEL HELPERS ────────────────────────────────────────────────────
def train_chronos(ts_uni, label):
    print(f"    - Chronos-2 for {label}...")
    ctx_len = min(len(ts_uni) - OUTPUT_CHUNK, 512)
    model = Chronos2Model(input_chunk_length=ctx_len, output_chunk_length=OUTPUT_CHUNK, hub_model_name=CHRONOS_MODEL)
    model.fit(ts_uni)
    return model.predict(OUTPUT_CHUNK, num_samples=1)

def train_nhits(target_uni, past_sc):
    print(f"    - NHiTS for {target_uni.components[0]}...")
    model = NHiTSModel(input_chunk_length=INPUT_CHUNK, output_chunk_length=OUTPUT_CHUNK, num_stacks=3, batch_size=BATCH_SIZE, n_epochs=MAX_EPOCHS,
                       pl_trainer_kwargs={"accelerator":"gpu","devices":1,"callbacks":[EarlyStopping("val_loss", patience=PATIENCE)]}, random_state=42)
    model.fit(target_uni[:-OUTPUT_CHUNK], past_covariates=past_sc[:-OUTPUT_CHUNK], val_series=target_uni[-INPUT_CHUNK-OUTPUT_CHUNK:], val_past_covariates=past_sc[-INPUT_CHUNK-OUTPUT_CHUNK:], verbose=False)
    pred = model.predict(OUTPUT_CHUNK, series=target_uni, past_covariates=past_sc, num_samples=1) # Point
    torch.cuda.empty_cache(); return pred

def train_stats(target_uni):
    print(f"    - Stats (ETS + Theta) for {target_uni.components[0]}...")
    # ETS
    ets = ExponentialSmoothing(trend=ModelMode.ADDITIVE, damped=True, seasonal=None)
    ets.fit(target_uni[:-OUTPUT_CHUNK])
    p_ets = ets.predict(OUTPUT_CHUNK, num_samples=1)
    # Theta (seasonality=1 for short data)
    th = Theta(seasonality_period=1)
    th.fit(target_uni[:-OUTPUT_CHUNK])
    p_th = th.predict(OUTPUT_CHUNK, num_samples=1)
    # FourTheta
    fth = FourTheta(seasonality_period=1)
    fth.fit(target_uni[:-OUTPUT_CHUNK])
    p_fth = fth.predict(OUTPUT_CHUNK, num_samples=1)
    return p_ets, p_th, p_fth


# ─── TAHAP 3: ENSEMBLE & RECONCILIATION ────────────────────────────────────────
def get_adaptive_blend(preds, val_sc):
    weights = []
    for p in preds:
        try:
            pv = p.values().flatten()
            av = val_sc.values().flatten()
            m = av > 1e-6
            err = np.mean(np.abs((av[m] - pv[m]) / av[m])) if m.any() else 1.0
            weights.append(1.0 / max(err, 0.01))
        except: weights.append(0.01)
    w = np.array(weights) / sum(weights)
    blended = sum(w[i] * preds[i].values() for i in range(len(preds)))
    return TimeSeries.from_times_and_values(preds[0].time_index, blended, freq="MS", columns=preds[0].components)

def trimmed_mean(ts, lo=10, hi=90):
    v = ts.all_values()
    l, h = np.percentile(v, lo, axis=2), np.percentile(v, hi, axis=2)
    m = np.where((v >= l[:,:,None]) & (v <= h[:,:,None]), v, np.nan)
    return np.nanmean(m, axis=2).flatten()


# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("[v4] Starting Trinity Ultra-Ensemble...")
    monthly = build_monthly_agg(KLAIM_PATH, POLIS_PATH)
    m_fe = engineer_features(monthly)
    
    # Prep Series
    past_cols = ["Prop_Inpatient", "Avg_Usia", "sin_month", "cos_month", "is_Q4", "roll3_f", "roll3_t", "yoy_f", "Time_Idx"]
    past_sc = Scaler().fit_transform(TimeSeries.from_times_and_values(m_fe.index, m_fe[past_cols].values, freq="MS"))
    
    targets = ["Frekuensi", "Severity", "Total_Claim"]
    scalers = {t: Scaler() for t in targets}
    series = {t: scalers[t].fit_transform(TimeSeries.from_times_and_values(m_fe.index, m_fe[t].values.reshape(-1,1), freq="MS", columns=[t])) for t in targets}
    
    final_preds = {}
    for t in targets:
        print(f"\n[Path] Processing {t}...")
        c2 = train_chronos(series[t], t)
        nh = train_nhits(series[t], past_sc)
        ets, th, fth = train_stats(series[t])
        
        # Ensemble Mix
        ens = get_adaptive_blend([nh, ets, th, fth], series[t][-OUTPUT_CHUNK:])
        
        # Blend Chronos-2 (60%) + DL/Stats Ensemble (40%)
        # All are point forecasts
        c2_v = c2.values().flatten()
        ens_v = ens.values().flatten()
        final_v = 0.6 * c2_v + 0.4 * ens_v
        
        # Unscale using Darts Scaler
        res_ts = TimeSeries.from_times_and_values(c2.time_index, final_v.reshape(-1,1), freq=c2.freq_str, columns=[t])
        final_preds[t] = scalers[t].inverse_transform(res_ts).values().flatten()

    # ── Reconciliation Step
    print("\n[Stage] Global Reconciliation (T = F * S)...")
    F, S, T = final_preds["Frekuensi"], final_preds["Severity"], final_preds["Total_Claim"]
    
    # Derived T from F*S
    T_from_FS = F * S
    # Balanced T: 50% direct, 50% structural
    T_final = 0.5 * T + 0.5 * T_from_FS
    
    # Adjust F and S slightly to match the new T balance
    # (Optional, but helps maintain consistency)
    F_final = 0.8 * F + 0.2 * (T_final / np.clip(S, 1, None))
    S_final = T_final / np.clip(F_final, 1, None)

    # ── Export
    rows = []
    dates = pd.date_range("2025-08-01", periods=5, freq="MS")
    for i, d in enumerate(dates):
        ms = d.strftime("%Y-%m")
        rows.extend([
            {"id": f"{ms}_Claim_Frequency", "value": round(float(F_final[i]))},
            {"id": f"{ms}_Claim_Severity",  "value": round(float(S_final[i]), 2)},
            {"id": f"{ms}_Total_Claim",     "value": round(float(T_final[i]), 2)},
        ])
    
    submission = pd.DataFrame(rows)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[v4] Final submission saved to {OUTPUT_PATH}")
    print(submission.tail(6))

if __name__ == "__main__":
    main()
