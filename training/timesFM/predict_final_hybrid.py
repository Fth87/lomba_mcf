
# Note: Requires Python 3.10-3.11 with 'timesfm[torch]' and 'statsmodels'
import pandas as pd
import numpy as np
import torch
import timesfm
import os
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("high")

# ==========================================
# 0. DATA LOADING
# ==========================================
print("\n--- FINAL HYBRID PREDICTION: V3 (Frequency) + TimesFM Robust (Value) ---")

try:
    df_polis = pd.read_csv("dataset/Data_Polis.csv")
    df_klaim = pd.read_csv("dataset/Data_Klaim.csv")
except FileNotFoundError:
    try:
        df_polis = pd.read_csv("/kaggle/input/datasets/fatihfawwaz/dataset-mcf/Data_Polis.csv")
        df_klaim = pd.read_csv("/kaggle/input/datasets/fatihfawwaz/dataset-mcf/Data_Klaim.csv")
    except:
            df_polis = pd.read_csv("Data_Polis.csv")
            df_klaim = pd.read_csv("Data_Klaim.csv") # Last attempt

df = pd.merge(df_klaim, df_polis, on="Nomor Polis", how="left")
for col in ["Tanggal Pembayaran Klaim", "Tanggal Pasien Masuk RS", "Tanggal Pasien Keluar RS", "Tanggal Efektif Polis"]:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Filter Full Dataset (Jan 2024 - July 2025)
df_filtered = df[(df["Tanggal Pasien Masuk RS"] >= "2024-01-01") & 
                    (df["Tanggal Pasien Masuk RS"] <= "2025-07-31")].copy()

df_filtered["Bulan"] = df_filtered["Tanggal Pasien Masuk RS"].dt.to_period("M")
monthly_agg = df_filtered.groupby("Bulan").agg(
    Frequency=("Claim ID", "nunique"),
    Total_Claim=("Nominal Klaim Yang Disetujui", "sum")
).reset_index()

monthly_agg["Severity"] = monthly_agg["Total_Claim"] / monthly_agg["Frequency"]
monthly_agg["Bulan_Ts"] = monthly_agg["Bulan"].dt.to_timestamp()
monthly_agg = monthly_agg.sort_values("Bulan_Ts")

print(f"Total Months: {len(monthly_agg)}")
print(monthly_agg.tail())

HORIZON = 5 # Aug, Sep, Oct, Nov, Dec 2025

# ==========================================
# 1. FREQUENCY FORECAST (CHAMPION: V3 Pipeline)
# ==========================================
print("\nPredicting Frequency with V3 (ETS + SES + SARIMAX)...")

def winsorize_series(series, lower=0.05, upper=0.95):
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

def forecast_ets_damped(series, steps):
    try:
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True).fit(optimized=True)
        return model.forecast(steps)
    except: return pd.Series([series.mean()]*steps)

def forecast_ses(series, steps):
    try:
        model = SimpleExpSmoothing(series).fit(optimized=True)
        return model.forecast(steps)
    except: return pd.Series([series.mean()]*steps)

def forecast_sarimax(series, steps):
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return model.forecast(steps)
    except: return pd.Series([series.mean()]*steps)

def robust_ensemble(series, steps):
    f_ses = forecast_ses(series, steps)
    f_ets = forecast_ets_damped(series, steps)
    f_sar = forecast_sarimax(series, steps)
    
    mean_val = series.mean()
    f_ses = f_ses.fillna(mean_val)
    f_ets = f_ets.fillna(mean_val)
    f_sar = f_sar.fillna(mean_val)
    
    # 40% SES, 40% ETS, 20% SARIMAX
    return (0.4 * f_ses.values + 0.4 * f_ets.values + 0.2 * f_sar.values)

def v3_pipeline(series, steps):
    work = series.copy()
    if work.isnull().any(): work = work.fillna(0)
    work = winsorize_series(work, upper=0.90)
    log_series = np.log1p(work)
    log_forecast = robust_ensemble(log_series, steps)
    return np.expm1(log_forecast)

final_freq_pred = v3_pipeline(monthly_agg["Frequency"], HORIZON)

# ==========================================
# 2. TOTAL CLAIM & SEVERITY FORECAST (CHAMPION: TimesFM Robust)
# ==========================================
print("Predicting Values with TimesFM Robust (Winsor + Log)...")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
config = timesfm.ForecastConfig(
    max_context=128, max_horizon=HORIZON, normalize_inputs=True,
    use_continuous_quantile_head=True, fix_quantile_crossing=True, infer_is_positive=True,
)
model.compile(config)

train_inputs_raw = [monthly_agg["Total_Claim"].values, monthly_agg["Severity"].values]
train_inputs_robust = []

for s in train_inputs_raw:
    w = winsorize_series(pd.Series(s), upper=0.90).values
    l = np.log1p(w)
    train_inputs_robust.append(l)

_, q_robust = model.forecast(horizon=HORIZON, inputs=train_inputs_robust)

# TimesFM Forecasts (Log -> Exp)
tfm_robust_total = np.expm1(q_robust[0, :, 5]) # Median
tfm_robust_sev = np.expm1(q_robust[1, :, 5])   # Median

# ==========================================
# 3. SUBMISSION ASSEMBLY
# ==========================================
print("\nStats:")
print(f"Freq (V3): {final_freq_pred}")
print(f"Total (TFM): {tfm_robust_total}")
print(f"Sev (TFM): {tfm_robust_sev}")

submission_list = []
forecast_dates = pd.date_range(start="2025-08-01", periods=HORIZON, freq="MS")

# We use the designated CHAMPION for each column
for i, date in enumerate(forecast_dates):
    date_str = date.strftime("%Y_%m")
    
    # Frequency -> V3
    val_freq = final_freq_pred[i]
    submission_list.append({"id": f"{date_str}_Claim_Frequency", "value": round(val_freq)})
    
    # Severity -> TimesFM Robust
    val_sev = tfm_robust_sev[i]
    submission_list.append({"id": f"{date_str}_Claim_Severity", "value": round(val_sev, 2)})

    # Total Claim -> TimesFM Robust
    val_total = tfm_robust_total[i]
    submission_list.append({"id": f"{date_str}_Total_Claim", "value": round(val_total, 2)})

submission_df = pd.DataFrame(submission_list)
output_dir = "submission"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/submission_final_hybrid.csv"
submission_df.to_csv(output_path, index=False)

print(f"\nFinal Hybrid Submission saved to {output_path}")
print(submission_df)
