
# Note: Requires Python 3.10-3.11 with 'timesfm[torch]' installed.
import pandas as pd
import numpy as np
import torch
import timesfm
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    torch.set_float32_matmul_precision("high")

# ==========================================
# 0. ROBUST HELPERS (Adapted from V3)
# ==========================================
def winsorize_series(series, lower=0.05, upper=0.95):
    """Cap outliers at percentiles to prevent model overreaction."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)

# ==========================================
# 1. Loading & Preprocessing Data
# ==========================================
print("Loading datasets...")
try:
    df_polis = pd.read_csv("/kaggle/input/datasets/fatihfawwaz/dataset-mcf/Data_Polis.csv")
    df_klaim = pd.read_csv("/kaggle/input/datasets/fatihfawwaz/dataset-mcf/Data_Klaim.csv")
except FileNotFoundError:
    df_polis = pd.read_csv("dataset/Data_Polis.csv")
    df_klaim = pd.read_csv("dataset/Data_Klaim.csv")

print("Merging and cleaning data...")
# Merge
df = pd.merge(df_klaim, df_polis, on="Nomor Polis", how="left")

# Date conversions
date_columns = ["Tanggal Pembayaran Klaim", "Tanggal Pasien Masuk RS", "Tanggal Pasien Keluar RS", "Tanggal Efektif Polis"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Filter Data (Jan 2024 - July 2025 as per context)
start_date = "2024-01-01"
end_date = "2025-07-31"
df_filtered = df[(df["Tanggal Pasien Masuk RS"] >= start_date) & 
                 (df["Tanggal Pasien Masuk RS"] <= end_date)].copy()

print(f"Total Claims after filtering: {len(df_filtered)}")

# Aggregate by Month
df_filtered["Bulan"] = df_filtered["Tanggal Pasien Masuk RS"].dt.to_period("M")
monthly_agg = df_filtered.groupby("Bulan").agg(
    Frequency=("Claim ID", "nunique"),
    Total_Claim=("Nominal Klaim Yang Disetujui", "sum")
).reset_index()

# Calculate Severity
monthly_agg["Severity"] = monthly_agg["Total_Claim"] / monthly_agg["Frequency"]
monthly_agg["Bulan_Ts"] = monthly_agg["Bulan"].dt.to_timestamp()
monthly_agg = monthly_agg.sort_values("Bulan_Ts")

print("Historical Data (Last 5 rows):")
print(monthly_agg.tail())

# ==========================================
# 2. TimesFM Forecasting (ROBUST PIPELINE)
# ==========================================
print("\nInitializing TimesFM 2.5 (200M) Model...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

print("Compiling model...")
config = timesfm.ForecastConfig(
    max_context=128,
    max_horizon=5,
    normalize_inputs=True,
    use_continuous_quantile_head=True, 
    fix_quantile_crossing=True,   
    infer_is_positive=True,  
)
model.compile(config)

series_names = ["Frequency", "Severity", "Total_Claim"]
horizon = 5

print(f"Forecasting {horizon} months ahead (ROBUST MODE: Winsorize -> Log -> TimesFM -> Exp)...")

# Prepare inputs
input_series = []

for name in series_names:
    original = monthly_agg[name].copy()
    
    # 1. Winsorize (Robustness)
    # Applying V3 logic: cap top 10% (upper=0.90) for strict outlier control
    processed = winsorize_series(original, upper=0.90) 
    
    # 2. Log Transform (Stabilize Variance)
    processed = np.log1p(processed)
    
    input_series.append(processed.values)

# Forecast on LOG SPACE
try:
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=input_series
    )
    # Using Median Forecast (Index 5) on LOG SPACE
    log_forecast_values = quantile_forecast[:, :, 5] 
    
    # 3. Inverse Transform (Exp)
    final_forecast_values = np.expm1(log_forecast_values)

except Exception as e:
    print(f"Error during forecast: {e}")
    raise e

# ==========================================
# 3. Submission Formatting
# ==========================================
print("Formatting submission...")

submission_list = []
forecast_dates = pd.date_range(start="2025-08-01", periods=horizon, freq="MS")

for i, date in enumerate(forecast_dates):
    date_str = date.strftime("%Y_%m")
    
    for idx, target in enumerate(series_names):
        val = final_forecast_values[idx, i]
        
        id_suffix = target
        if target == "Frequency": id_suffix = "Claim_Frequency"
        elif target == "Severity": id_suffix = "Claim_Severity"
        
        row_id = f"{date_str}_{id_suffix}"
        
        submission_list.append({
            "id": row_id,
            "value": max(0, val) # Ensure no negative predictions
        })

submission_df = pd.DataFrame(submission_list)
output_dir = "submission"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/submission_timesfm_robust.csv"
submission_df.to_csv(output_path, index=False)

print(f"Robust Submission saved to {output_path}")
print(submission_df)
