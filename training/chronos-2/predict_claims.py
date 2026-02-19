
# %%
import pandas as pd
import numpy as np
from darts import TimeSeries, concatenate
# from darts.models import Chronos # Failed import
from darts.models import Chronos2Model
from darts.dataprocessing.transformers import Scaler
import warnings
import os
import torch

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Tahap 1: Persiapan dan Penggabungan Data

# %%
# Load Datasets
print("Loading datasets...")
df_polis = pd.read_csv("dataset/Data_Polis.csv")
df_klaim = pd.read_csv("dataset/Data_Klaim.csv")

# Identify Keys & Merge
print("Merging data...")
df = pd.merge(df_klaim, df_polis, on="Nomor Polis", how="left")

# Data Cleaning (Dates)
print("Cleaning dates...")
date_columns = ["Tanggal Pembayaran Klaim", "Tanggal Pasien Masuk RS", "Tanggal Pasien Keluar RS", "Tanggal Efektif Polis"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df["Tanggal Lahir"] = pd.to_datetime(df["Tanggal Lahir"], format="%Y%m%d", errors='coerce')

# Analysis Population (Filter Date Range)
start_date = "2024-01-01"
end_date = "2025-07-31"
df_filtered = df[(df["Tanggal Pasien Masuk RS"] >= start_date) & 
                 (df["Tanggal Pasien Masuk RS"] <= end_date)].copy()

print(f"Total Claims after filtering: {len(df_filtered)}")

# %% [markdown]
# ## Tahap 2: Feature Engineering (Rekayasa Fitur)

# %%
# Profil Risiko: Usia & Wilayah
reference_date = pd.to_datetime("2025-08-01")
df_filtered["Usia"] = (reference_date - df_filtered["Tanggal Lahir"]).dt.days // 365
df_filtered["Wilayah"] = df_filtered["Plan Code"].map({"M-001": "Wilayah A", "M-002": "Wilayah B", "M-003": "Wilayah C"})

# Kategorisasi Medis: ICD Grouping
df_filtered["ICD_Group"] = df_filtered["ICD Diagnosis"].astype(str).str.split('.').str[0]

# Durasi dan Metode
df_filtered["Durasi_Rawat"] = (df_filtered["Tanggal Pasien Keluar RS"] - df_filtered["Tanggal Pasien Masuk RS"]).dt.days
df_filtered["Durasi_Rawat"] = df_filtered["Durasi_Rawat"].clip(lower=0)

# %% [markdown]
# ## Tahap 3: Perhitungan Target Historis
# Aggregating by Month for Forecasting

# %%
df_filtered["Bulan"] = df_filtered["Tanggal Pasien Masuk RS"].dt.to_period("M")

monthly_agg = df_filtered.groupby("Bulan").agg(
    Frequency=("Claim ID", "nunique"),
    Total_Claim=("Nominal Klaim Yang Disetujui", "sum")
).reset_index()

monthly_agg["Severity"] = monthly_agg["Total_Claim"] / monthly_agg["Frequency"]
monthly_agg["Bulan_Ts"] = monthly_agg["Bulan"].dt.to_timestamp()

print("Historical Metrics (First 5 rows):")
print(monthly_agg.head())

# %% [markdown]
# ## Tahap 4: Pengembangan Model Prediksi (Darts Chronos-2)

# %%
ts_freq = TimeSeries.from_dataframe(monthly_agg, "Bulan_Ts", "Frequency")
ts_total = TimeSeries.from_dataframe(monthly_agg, "Bulan_Ts", "Total_Claim")
ts_sev = TimeSeries.from_dataframe(monthly_agg, "Bulan_Ts", "Severity")

# Chronos Model Setup
# Using 'autogluon/chronos-2-small' (28M params) which is a true Chronos-2 model.
model_name = "autogluon/chronos-2" 
input_chunk_length = 12 # Look back 12 months
output_chunk_length = 5 # Forecast 5 months

print(f"Initializing Chronos2Model models: {model_name}")

def forecast_series(series, model_name, input_len, output_len):
    # Depending on the installed version, 'chronos-2' might be default, but we enforce 'amazon/chronos-t5-tiny'
    model = Chronos2Model(
        input_chunk_length=input_len,
        output_chunk_length=output_len,
        hub_model_name=model_name,
        random_state=42
    )
    model.fit(series)
    prediction = model.predict(output_len)
    return prediction

print("Forecasting Frequency...")
pred_freq = forecast_series(ts_freq, model_name, input_chunk_length, output_chunk_length)

print("Forecasting Total Claim...")
pred_total = forecast_series(ts_total, model_name, input_chunk_length, output_chunk_length)

print("Forecasting Severity...")
pred_sev = forecast_series(ts_sev, model_name, input_chunk_length, output_chunk_length)

print("Forecasts generated.")

# %% [markdown]
# ## Tahap 5: Penyusunan Submisi Akhir

# %%
submission_list = []
forecast_dates = pd.date_range(start="2025-08-01", periods=5, freq="MS")

freq_values = pred_freq.values()
total_values = pred_total.values()
sev_values = pred_sev.values()

for i, date in enumerate(forecast_dates):
    date_str = date.strftime("%Y_%m")
    
    submission_list.append({
        "id": f"{date_str}_Claim_Frequency",
        "value": freq_values[i][0]
    })
    
    submission_list.append({
        "id": f"{date_str}_Claim_Severity",
        "value": sev_values[i][0]
    })
    
    submission_list.append({
        "id": f"{date_str}_Total_Claim",
        "value": total_values[i][0]
    })

submission_df = pd.DataFrame(submission_list)

# Verify rows
if len(submission_df) != 15:
    print(f"Warning: Expected 15 rows, got {len(submission_df)}")

# Output
output_path = "submission/submission_chronos2.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
print(submission_df)
