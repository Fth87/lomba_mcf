
# Note: This script requires 'timesfm' library which currently supports Python 3.10 - 3.11.
# It may not install/run on Python 3.12+ (as of early 2026).
# Recommended environment: Python 3.10 with `pip install "timesfm[torch]"`

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
# 1. Loading & Preprocessing Data
# ==========================================
print("Loading datasets...")
try:
    df_polis = pd.read_csv("/kaggle/input/datasets/fatihfawwaz/dataset-mcf/Data_Polis.csv")
    df_klaim = pd.read_csv("/kaggle/input/datasets/fatihfawwaz/dataset-mcf/Data_Klaim.csv")
except FileNotFoundError:
    # Adjust path if running from a different directory
    df_polis = pd.read_csv("../dataset/Data_Polis.csv")
    df_klaim = pd.read_csv("../dataset/Data_Klaim.csv")

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

# Feature Engineering: Simple Date Covariates (as requested)2
# Although TimesFM is a foundation model that learns temporal patterns well on its own,
# explicit features can sometimes help with seasonality.
monthly_agg["Month"] = monthly_agg["Bulan_Ts"].dt.month
monthly_agg["Quarter"] = monthly_agg["Bulan_Ts"].dt.quarter
# Note: For this simple script, we are not passing these as external regressors yet 
# because it requires more complex tensor setup. 
# But this prepares the dataframe for it.

# Sort by date
monthly_agg = monthly_agg.sort_values("Bulan_Ts")

print("Historical Data (Last 5 rows):")
print(monthly_agg.tail())

# ==========================================
# 2. TimesFM Forecasting
# ==========================================
print("\nInitializing TimesFM 2.5 (200M) Model...")

# Initialize model
# Note: The 2.5 PyTorch model wrapper handles device placement internally or defaults to CPU/CUDA if available.
# We do not need to manually call .to(device) on the wrapper object.
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

# Compile the model with configuration
# Compile the model with configuration
print("Compiling model...")
config = timesfm.ForecastConfig(
    max_context=128,          # Adjust based on your input length
    max_horizon=5,            # Forecasting 5 steps provided in plan
    normalize_inputs=True,
    use_continuous_quantile_head=True, # Improved quantile estimation
    fix_quantile_crossing=True,        # Ensure quantiles are ordered
    infer_is_positive=True,            # Claims can't be negative
)
model.compile(config)

# Prepare inputs for forecasting
# TimesFM expects a list of numpy arrays for univariate forecasting
# We have 3 series: Frequency, Total_Claim, Severity
series_dict = {
    "Frequency": monthly_agg["Frequency"].values,
    "Severity": monthly_agg["Severity"].values,
    "Total_Claim": monthly_agg["Total_Claim"].values
}

input_series = [
    series_dict["Frequency"],
    series_dict["Severity"],
    series_dict["Total_Claim"]
]

horizon = 5 # Predict Aug, Sep, Oct, Nov, Dec 2025

print(f"Forecasting {horizon} months ahead...")

# Forecast
try:
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=input_series
    )
    # Strategy for MAPE Improvement: Use Quintile 50 (Median) instead of Mean
    # quantile_forecast shape: (batch_size, horizon, num_quantiles)
    # Default quantiles: 10th - 90th percentile (10 steps)
    # Index 5 corresponds to the 50th percentile (Median)
    forecast_values = quantile_forecast[:, :, 5] 
    
except Exception as e:
    print(f"Error during forecast: {e}")
    # Fallback/Debug info would go here
    raise e

# ==========================================
# 3. Submission Formatting
# ==========================================
print("Formatting submission...")

submission_list = []
forecast_dates = pd.date_range(start="2025-08-01", periods=horizon, freq="MS")

# Map back indices to names
target_names = ["Frequency", "Severity", "Total_Claim"]

for i, date in enumerate(forecast_dates):
    date_str = date.strftime("%Y_%m") # Format: 2025_08
    
    for idx, target in enumerate(target_names):
        # Value for this month and this target
        val = forecast_values[idx, i]
        
        # ID Format: 2025_08_Claim_Frequency
        id_suffix = target
        if target == "Frequency":
            id_suffix = "Claim_Frequency"
        elif target == "Severity":
            id_suffix = "Claim_Severity"
        
        row_id = f"{date_str}_{id_suffix}"
        
        submission_list.append({
            "id": row_id,
            "value": max(0, val) # Ensure no negative predictions
        })


submission_df = pd.DataFrame(submission_list)

# Output
output_dir = "submission"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/submission_timesfm.csv"
submission_df.to_csv(output_path, index=False)

print(f"Submission saved to {output_path}")
print(submission_df)
