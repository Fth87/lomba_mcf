
# Note: Requires Python 3.10-3.11 for TimesFM execution.
import pandas as pd
import numpy as np
import torch
import timesfm
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("high")

# ==========================================
# 1. Loading & Splitting
# ==========================================
print("\n--- Benchmark: Simple vs Robust TimesFM ---")

def load_data():
    try:
        df_polis = pd.read_csv("Data_Polis.csv")
        df_klaim = pd.read_csv("Data_Klaim.csv")
    except FileNotFoundError:
        df_polis = pd.read_csv("dataset/Data_Polis.csv")
        df_klaim = pd.read_csv("dataset/Data_Klaim.csv")

    df = pd.merge(df_klaim, df_polis, on="Nomor Polis", how="left")
    
    date_columns = ["Tanggal Pembayaran Klaim", "Tanggal Pasien Masuk RS", "Tanggal Pasien Keluar RS", "Tanggal Efektif Polis"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

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
    return monthly_agg

monthly_agg = load_data()

# Test Set: Last 5 months (March 2025 - July 2025)
TEST_SIZE = 5
train_df = monthly_agg.iloc[:-TEST_SIZE].copy()
test_df = monthly_agg.iloc[-TEST_SIZE:].copy()

print(f"\nTrain Set: {len(train_df)} months (Jan 2024 - Feb 2025)")
print(f"Test Set: {len(test_df)} months (March 2025 - July 2025)")

series_names = ["Frequency", "Severity", "Total_Claim"]
raw_actuals = [test_df[name].values for name in series_names]
train_values = [train_df[name].values for name in series_names]


# ==========================================
# 2. Forecasting
# ==========================================
print("\n--- Initializing TimesFM Model ---")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
config = timesfm.ForecastConfig(
    max_context=128, max_horizon=TEST_SIZE, normalize_inputs=True,
    use_continuous_quantile_head=True, fix_quantile_crossing=True, infer_is_positive=True,
)
model.compile(config)

# A. SIMPLE Method (Raw -> TimesFM -> Raw)
print("Running SIMPLE TimesFM...")
simple_input = train_values
_, q_simple = model.forecast(horizon=TEST_SIZE, inputs=simple_input)
simple_forecast = q_simple[:, :, 5] # Median

# B. ROBUST Method (Winsor -> Log -> TimesFM -> Exp)
print("Running ROBUST TimesFM (Winsor + Log)...")

def winsorize(series, upper=0.90):
    series = pd.Series(series)
    upper_bound = series.quantile(upper)
    return series.clip(upper=upper_bound).values

robust_input = []
for series in train_values:
    # 1. Winsorize
    w = winsorize(series, upper=0.90) 
    # 2. Log
    l = np.log1p(w)
    robust_input.append(l)

_, q_robust = model.forecast(horizon=TEST_SIZE, inputs=robust_input)
robust_forecast_log = q_robust[:, :, 5] # Median on Log
# 3. Exp
robust_forecast = np.expm1(robust_forecast_log)

# ==========================================
# 3. Evaluation
# ==========================================
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print("\n--- Benchmark Results (MAPE %) ---")

results = []
for idx, name in enumerate(series_names):
    mape_simple = calculate_mape(raw_actuals[idx], simple_forecast[idx])
    mape_robust = calculate_mape(raw_actuals[idx], robust_forecast[idx])
    
    results.append({
        "Target": name,
        "Simple MAPE": mape_simple,
        "Robust MAPE": mape_robust,
        "Gain": mape_simple - mape_robust
    })

results_df = pd.DataFrame(results)
print(results_df.round(2))

avg_simple = results_df["Simple MAPE"].mean()
avg_robust = results_df["Robust MAPE"].mean()

print("\n--- Summary ---")
print(f"Average Simple MAPE: {avg_simple:.2f}%")
print(f"Average Robust MAPE: {avg_robust:.2f}%")

if avg_robust < avg_simple:
    print("Result: ROBUST approach (Winsor + Log) is BETTER.")
else:
    print("Result: SIMPLE approach is BETTER.")
