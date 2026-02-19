"""
=============================================================================
Forecast Pipeline V3: Robustness & Generalization
=============================================================================
Strategy to combat Overfitting (Ground Truth MAPE 10.77% vs Val 6.78%):
1. Outlier Handling: Winsorization (Capping extreme spikes)
2. Trend Dampening: Use Damped ETS & Simple Exponential Smoothing (SES)
3. Robust Validation: Rolling Cross-Validation (3 folds)
4. Conservative Ensemble: SES (40%) + Damped ETS (40%) + SARIMAX (20%)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================================================================
# 1. LOAD & PREPROCESSING
# ============================================================================
print("STEP 1: LOAD & PREPROCESSING")
df_klaim = pd.read_csv("Data_Klaim.csv")
df_polis = pd.read_csv("Data_Polis.csv")

# Date Conversions
cols_klaim = ["Tanggal Pembayaran Klaim", "Tanggal Pasien Masuk RS", "Tanggal Pasien Keluar RS"]
for col in cols_klaim:
    df_klaim[col] = pd.to_datetime(df_klaim[col], errors="coerce")
for col in ["Tanggal Lahir", "Tanggal Efektif Polis"]:
    df_polis[col] = pd.to_datetime(df_polis[col].astype(str), format="%Y%m%d", errors="coerce")

# Merge
df = df_klaim.merge(df_polis, on="Nomor Polis", how="left")
df = df.dropna(subset=["Tanggal Pasien Masuk RS"])

# Aggregate
df["YearMonth"] = df["Tanggal Pasien Masuk RS"].dt.to_period("M")
monthly = df.groupby("YearMonth").agg(
    Frequency=("Claim ID", "count"),
    Total_Claim=("Nominal Klaim Yang Disetujui", "sum"),
).reset_index()

monthly["Severity"] = monthly["Total_Claim"] / monthly["Frequency"]
monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()
monthly = monthly.sort_values("Date").set_index("Date").asfreq("MS").fillna(0)

print(f"Data range: {monthly.index.min()} to {monthly.index.max()}")

# ============================================================================
# 2. ROBUSTNESS FUNCTIONS
# ============================================================================

def winsorize_series(series, lower=0.05, upper=0.95):
    """Cap outliers at percentiles to prevent model overreaction."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

def mape(actual, predicted):
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # Ensure equal length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# ============================================================================
# 3. MODELS
# ============================================================================

def forecast_ets_damped(series, steps):
    """ETS with Damped Trend (Conservative)"""
    try:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
        ).fit(optimized=True)
        pred = model.forecast(steps)
        if pred.isnull().any(): return pd.Series([series.mean()]*steps)
        return pred
    except:
        return pd.Series([series.mean()]*steps)

def forecast_ses(series, steps):
    """Simple Exponential Smoothing (Level Only, No Trend) - Very Robust"""
    try:
        model = SimpleExpSmoothing(series).fit(optimized=True)
        pred = model.forecast(steps)
        if pred.isnull().any(): return pd.Series([series.mean()]*steps)
        return pred
    except:
        return pd.Series([series.mean()]*steps)

def forecast_sarimax(series, steps):
    """SARIMAX (1,1,1)x(1,0,0,12) - Simpler seasonal component"""
    try:
        # Simplified seasonal order to avoid overfitting
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = model.forecast(steps)
        if pred.isnull().any(): return pd.Series([series.mean()]*steps)
        return pred
    except:
        return pd.Series([series.mean()]*steps)

def robust_ensemble(series, steps):
    """
    Weighted Ensemble:
    - 40% SES (Stability)
    - 40% ETS Damped (Trend awareness but conservative)
    - 20% SARIMAX (Pattern recognition)
    """
    f_ses = forecast_ses(series, steps)
    f_ets = forecast_ets_damped(series, steps)
    f_sar = forecast_sarimax(series, steps)
    
    # Check for NaNs and fallback
    mean_val = series.mean()
    if f_ses.isnull().any(): f_ses = f_ses.fillna(mean_val)
    if f_ets.isnull().any(): f_ets = f_ets.fillna(mean_val)
    if f_sar.isnull().any(): f_sar = f_sar.fillna(mean_val)
    
    # Force alignment (use SES index as reference)
    try:
        ensemble_vals = (0.4 * f_ses.values + 
                         0.4 * f_ets.values + 
                         0.2 * f_sar.values)
        return pd.Series(ensemble_vals, index=f_ses.index)
    except Exception as e:
        # Fallback if shapes mismatch drastically
        return pd.Series([mean_val]*steps, index=f_ses.index)

def forecast_pipeline(series, steps, winsor=True):
    """Full Pipeline: Winsorize -> Log -> Forecast -> Exp"""
    work_series = series.copy()
    
    # Check initial
    if work_series.isnull().any():
         work_series = work_series.fillna(0)
    
    # 1. Winsorize (Handle Spikes)
    if winsor:
        work_series = winsorize_series(work_series, upper=0.90) # Cap top 10%
        
    # 2. Log Transform
    log_series = np.log1p(work_series)
    if not np.isfinite(log_series).all():
        log_series = log_series.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Forecast
    log_forecast = robust_ensemble(log_series, steps)
    
    # 4. Inverse Transform
    final_pred = np.expm1(log_forecast)
    return final_pred

# ============================================================================
# 4. ROLLING CROSS-VALIDATION
# ============================================================================
print("\n" + "="*50)
print("STEP 2: ROLLING CROSS-VALIDATION (3 Folds)")
print("="*50)

# Folds:
# 1. Train up to Apr 2025, Test May 2025
# 2. Train up to May 2025, Test Jun 2025
# 3. Train up to Jun 2025, Test Jul 2025

folds = [
    (pd.Timestamp("2025-05-01"), 1),
    (pd.Timestamp("2025-06-01"), 1),
    (pd.Timestamp("2025-07-01"), 1)
]

scores = {"Frequency": [], "Total_Claim": [], "Severity": []}

for cutoff, steps in folds:
    train = monthly[monthly.index < cutoff]
    test = monthly[(monthly.index >= cutoff) & (monthly.index < cutoff + pd.DateOffset(months=steps))]
    
    if len(test) == 0: continue
    
    print(f"\nFold Cutoff: {cutoff.date()} (Test: {test.index[0].date()})")
    
    # Forecast
    p_freq = forecast_pipeline(train["Frequency"], steps)
    p_total = forecast_pipeline(train["Total_Claim"], steps)
    p_sev = p_total / p_freq
    
    # Evaluate
    s_freq = mape(test["Frequency"], p_freq)
    s_total = mape(test["Total_Claim"], p_total)
    s_sev = mape(test["Severity"], p_sev)
    
    scores["Frequency"].append(s_freq)
    scores["Total_Claim"].append(s_total)
    scores["Severity"].append(s_sev)
    
    print(f"  Freq MAPE: {s_freq:.2f}%, Total MAPE: {s_total:.2f}%")

avg_freq = np.mean(scores["Frequency"])
avg_total = np.mean(scores["Total_Claim"])
avg_sev = np.mean(scores["Severity"])
final_cv_score = (avg_freq + avg_total + avg_sev) / 3

print("\n" + "-"*30)
print(f"CROSS-VALIDATION MAPE: {final_cv_score:.2f}%")
print(f"Freq: {avg_freq:.2f}%, Total: {avg_total:.2f}%, Sev: {avg_sev:.2f}%")
print("-"*30)

# ============================================================================
# 5. FINAL VISUALIZATION & SUBMISSION
# ============================================================================
print("\nSTEP 3: GENERATING FINAL FORECAST (v3)")

forecast_steps = 5
f_final_freq = forecast_pipeline(monthly["Frequency"], forecast_steps)
f_final_total = forecast_pipeline(monthly["Total_Claim"], forecast_steps)
f_final_sev = f_final_total / f_final_freq

# Generate Submission
forecast_months = pd.date_range("2025-08-01", periods=forecast_steps, freq="MS")
rows = []
for i, month in enumerate(forecast_months):
    month_str = month.strftime("%Y_%m")
    rows.append({"id": f"{month_str}_Claim_Frequency", "value": round(f_final_freq.iloc[i])})
    rows.append({"id": f"{month_str}_Claim_Severity", "value": round(f_final_sev.iloc[i], 2)})
    rows.append({"id": f"{month_str}_Total_Claim", "value": round(f_final_total.iloc[i], 2)})

submission = pd.DataFrame(rows)
submission.to_csv("submission_v3.csv", index=False)
print("\nSaved to submission_v3.csv")
print(submission.head(15).to_string(index=False))
