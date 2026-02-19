
# %%
import pandas as pd
from nixtla import NixtlaClient
from dotenv import load_dotenv
import warnings
import os

warnings.filterwarnings("ignore")
load_dotenv()

# %%
# Load Data & Merge
df_polis = pd.read_csv("dataset/Data_Polis.csv")
df_klaim = pd.read_csv("dataset/Data_Klaim.csv")

df = pd.merge(df_klaim, df_polis, on="Nomor Polis", how="left")

# Data Cleaning
date_cols = ["Tanggal Pembayaran Klaim", "Tanggal Pasien Masuk RS", "Tanggal Pasien Keluar RS", "Tanggal Efektif Polis"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df["Tanggal Lahir"] = pd.to_datetime(df["Tanggal Lahir"], format="%Y%m%d", errors='coerce')

# Filter Population
df = df[(df["Tanggal Pasien Masuk RS"] >= "2024-01-01") & 
        (df["Tanggal Pasien Masuk RS"] <= "2025-07-31")].copy()

# Feature Engineering
reference_date = pd.to_datetime("2025-08-01")
df["Usia"] = (reference_date - df["Tanggal Lahir"]).dt.days // 365
df["Wilayah"] = df["Plan Code"].map({"M-001": "Wilayah A", "M-002": "Wilayah B", "M-003": "Wilayah C"})
df["Durasi_Rawat"] = (df["Tanggal Pasien Keluar RS"] - df["Tanggal Pasien Masuk RS"]).dt.days.clip(lower=0)

# %%
# Aggregate Monthly
df["Bulan"] = df["Tanggal Pasien Masuk RS"].dt.to_period("M").dt.to_timestamp()

agg_df = df.groupby("Bulan").agg({
    "Claim ID": "nunique",
    "Nominal Klaim Yang Disetujui": "sum"
}).rename(columns={"Claim ID": "Frequency", "Nominal Klaim Yang Disetujui": "Total_Claim"})
agg_df["Severity"] = agg_df["Total_Claim"] / agg_df["Frequency"]
agg_df = agg_df.reset_index()

# Prepare for Nixtla (long format)
metrics = ["Frequency", "Total_Claim", "Severity"]
nixtla_df = pd.melt(agg_df, id_vars=["Bulan"], value_vars=metrics, var_name="unique_id", value_name="y")
nixtla_df = nixtla_df.rename(columns={"Bulan": "ds"})

# %%
# Initialize Nixtla Client (Default URL)
nixtla_client = NixtlaClient(
    api_key=os.getenv('nixtla_key')
)
nixtla_client.validate_api_key()

# Forecast
forecast_df = nixtla_client.forecast(
    df=nixtla_df,
    h=5,
    freq='MS',
    time_col='ds',
    target_col='y'
)

# %%
# Format Submission
submission_list = []
forecast_map = forecast_df.set_index(['ds', 'unique_id'])['TimeGPT']

dates = pd.date_range(start="2025-08-01", periods=5, freq="MS")

for date in dates:
    date_str = date.strftime("%Y_%m")
    
    # Frequency
    val_freq = forecast_map.get((date, "Frequency"), 0)
    submission_list.append({"id": f"{date_str}_Claim_Frequency", "value": val_freq})
    
    # Severity
    val_sev = forecast_map.get((date, "Severity"), 0)
    submission_list.append({"id": f"{date_str}_Claim_Severity", "value": val_sev})
    
    # Total Claim
    val_total = forecast_map.get((date, "Total_Claim"), 0)
    submission_list.append({"id": f"{date_str}_Total_Claim", "value": val_total})

sub_df = pd.DataFrame(submission_list)
sub_df.to_csv("submission/submission_nixla.csv", index=False)
print("Submission saved to submission.csv")
print(sub_df)
