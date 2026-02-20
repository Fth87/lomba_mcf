"""
TFT Pipeline — Multi-Horizon Insurance Claim Forecasting
Corrected version: exposure filter, identity normalizer, vectorized future
dataset (LOCF + exponential decay), proper h=5 backtesting, CUDA cleanup.
"""
import warnings
warnings.filterwarnings("ignore")

import gc
import argparse
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.metrics import TweedieLoss, PoissonLoss

KLAIM_PATH  = "dataset/Data_Klaim.csv"
POLIS_PATH  = "dataset/Data_Polis.csv"
OUTPUT_PATH = "submission/submission_tft.csv"

MAX_ENCODER  = 6
MAX_PRED     = 5
BATCH_SIZE   = 32
MAX_EPOCHS   = 30
PATIENCE     = 5
DECAY_ALPHA  = 0.7


# ─── TAHAP 1: PANEL DATA ────────────────────────────────────────────────────────
def build_panel(klaim_path: str, polis_path: str):
    df_k = pd.read_csv(klaim_path)
    df_p = pd.read_csv(polis_path)

    df_k["Tanggal Pasien Masuk RS"] = pd.to_datetime(
        df_k["Tanggal Pasien Masuk RS"], errors="coerce")
    df_p["Tanggal_Lahir"] = pd.to_datetime(
        df_p["Tanggal Lahir"].astype(str), format="%Y%m%d", errors="coerce")
    df_p["Tanggal_Efektif"] = pd.to_datetime(
        df_p["Tanggal Efektif Polis"].astype(str), format="%Y%m%d", errors="coerce")
    df_p = df_p.rename(columns={
        "Nomor Polis": "Nomor_Polis",
        "Plan Code":   "Plan_Code",
    })
    df_k = df_k.rename(columns={"Nomor Polis": "Nomor_Polis"})

    # p98 cap pada level individu
    p98 = df_k["Nominal Klaim Yang Disetujui"].quantile(0.98)
    df_k["Nominal_Final"] = df_k["Nominal Klaim Yang Disetujui"].clip(upper=p98)
    print(f"[1] p98 cap = {p98:,.0f}")

    df_k["ICD_Group"]    = df_k["ICD Diagnosis"].fillna("").str[:3].replace("", "UNK")
    df_k["is_inpatient"] = (df_k["Inpatient/Outpatient"].str.strip().str.upper() == "IP").astype(float)
    df_k["Bulan_Obs"]    = df_k["Tanggal Pasien Masuk RS"].dt.to_period("M").dt.to_timestamp()

    claim_agg = df_k.groupby(["Nomor_Polis", "Bulan_Obs"]).agg(
        Total_Claim_Individu  = ("Nominal_Final", "sum"),
        Frekuensi_Individu    = ("Nominal_Final", "count"),
        ICD_Group             = ("ICD_Group", lambda x: x.mode().iloc[0]),
        sum_ip                = ("is_inpatient", "sum"),
        cnt_total             = ("is_inpatient", "count"),
    ).reset_index()

    all_months = pd.DataFrame({"Bulan_Obs": pd.date_range("2024-01-01", "2025-07-01", freq="MS")})
    df_p["_k"] = 1; all_months["_k"] = 1
    panel = df_p.merge(all_months, on="_k").drop("_k", axis=1)

    # Exposure guard: drop structural zeros (polis belum efektif)
    panel = panel[panel["Bulan_Obs"] >= panel["Tanggal_Efektif"]].copy()
    print(f"[1] Panel rows (after exposure filter): {len(panel):,}")

    panel = panel.merge(claim_agg, on=["Nomor_Polis", "Bulan_Obs"], how="left")
    panel["Total_Claim_Individu"] = panel["Total_Claim_Individu"].fillna(0.0)
    panel["Frekuensi_Individu"]   = panel["Frekuensi_Individu"].fillna(0.0)
    panel["ICD_Group"]            = panel["ICD_Group"].fillna("UNK")
    panel["sum_ip"]               = panel["sum_ip"].fillna(0.0)
    panel["cnt_total"]            = panel["cnt_total"].fillna(0.0)

    return panel, df_p[["Nomor_Polis", "Tanggal_Lahir", "Tanggal_Efektif",
                         "Plan_Code", "Gender", "Domisili"]]


# ─── TAHAP 2: FEATURE ENGINEERING ──────────────────────────────────────────────
def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["Nomor_Polis", "Bulan_Obs"]).reset_index(drop=True)

    # Usia aktuaria presisi (vektorisasi, tanpa leap-year drift)
    dob = panel["Tanggal_Lahir"]
    obs = panel["Bulan_Obs"]
    panel["Usia"] = (obs.dt.year - dob.dt.year) - (
        (obs.dt.month < dob.dt.month)
        | ((obs.dt.month == dob.dt.month) & (obs.dt.day < dob.dt.day))
    ).astype(int)

    # Prop_Inpatient: rolling 3 bulan per polis
    panel["raw_prop"] = (panel["sum_ip"] / panel["cnt_total"].replace(0, np.nan)).fillna(0.0)
    panel["Prop_Inpatient"] = (
        panel.groupby("Nomor_Polis")["raw_prop"]
             .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Time_Idx (integer 0–18)
    months_sorted = sorted(panel["Bulan_Obs"].unique())
    m2i = {m: i for i, m in enumerate(months_sorted)}
    panel["Time_Idx"] = panel["Bulan_Obs"].map(m2i)

    for c in ["Plan_Code", "Gender", "Domisili", "ICD_Group"]:
        panel[c] = panel[c].astype("category")

    return panel.drop(columns=["raw_prop", "sum_ip", "cnt_total",
                                "Tanggal_Lahir", "Tanggal_Efektif"], errors="ignore")


# ─── TAHAP 3: TimeSeriesDataSet ─────────────────────────────────────────────────
def make_dataset(df: pd.DataFrame, target: str) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        df,
        group_ids                        = ["Nomor_Polis"],
        time_idx                         = "Time_Idx",
        target                           = target,
        max_encoder_length               = MAX_ENCODER,
        max_prediction_length            = MAX_PRED,
        static_categoricals              = ["Plan_Code", "Gender", "Domisili"],
        time_varying_known_reals         = ["Time_Idx", "Usia"],
        time_varying_unknown_reals       = [target, "Prop_Inpatient"],
        time_varying_unknown_categoricals= ["ICD_Group"],
        # Identity normalizer: Tweedie/Poisson bekerja pada skala absolut
        target_normalizer                = TorchNormalizer(method="identity", center=False),
        allow_missing_timesteps          = True,
        add_relative_time_idx            = True,
        add_target_scales                = False,
        add_encoder_length               = True,
    )


# ─── TAHAP 4: TRAINING + INFERENSI ─────────────────────────────────────────────
def train_and_infer(training_ds: TimeSeriesDataSet,
                    full_panel: pd.DataFrame,
                    last_tidx: int,
                    loss_fn,
                    target: str,
                    label: str):
    train_dl = training_ds.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    pred_ds = TimeSeriesDataSet.from_dataset(
        training_ds, full_panel, predict=True, stop_randomization=True)
    pred_dl = pred_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate        = 3e-3,
        hidden_size          = 32,
        attention_head_size  = 2,
        dropout              = 0.1,
        hidden_continuous_size = 16,
        loss                 = loss_fn,
        log_interval         = -1,
        reduce_on_plateau_patience = 0,   # disable — EarlyStopping handles regularization
    )
    print(f"[{label}] Parameters: {tft.size() / 1e3:.1f}k")

    trainer = Trainer(
        max_epochs          = MAX_EPOCHS,
        accelerator         = "gpu",
        devices             = 1,
        precision           = "32-true",
        gradient_clip_val   = 0.1,
        callbacks           = [EarlyStopping("val_loss", patience=PATIENCE, mode="min")],
        enable_progress_bar = True,
        enable_model_summary= False,
        logger              = False,
    )
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=pred_dl)

    result  = tft.predict(pred_dl, return_index=True, return_x=False)
    idx_df  = result.index.copy()       # contains sequence info, NOT direct Time_Idx values
    preds   = result.output             # (N_polis, MAX_PRED) — each col = 1 future horizon

    # Normalize to 2D (N_polis, MAX_PRED)
    if preds.ndim == 3:
        preds = preds[:, :, preds.shape[2] // 2]   # median quantile if quantile output
    preds_np  = preds.numpy()           # shape: (N_polis, MAX_PRED=5)
    n_polis   = preds_np.shape[0]

    # Retrieve ordered Nomor_Polis from full_panel using the sequence ordering
    # pred_ds sorts by Nomor_Polis (same order as full_panel groups)
    polis_order = (full_panel[full_panel["Time_Idx"] == last_tidx]
                   .sort_values("Nomor_Polis")["Nomor_Polis"]
                   .values[:n_polis])

    # future Time_Idx = [19, 20, 21, 22, 23] — known deterministically from build_future_panel
    future_tidx_vals = sorted(full_panel[full_panel["Time_Idx"] > last_tidx]["Time_Idx"].unique())

    long_rows = []
    for step_i, tidx_val in enumerate(future_tidx_vals):
        long_rows.append(pd.DataFrame({
            "Nomor_Polis": polis_order,
            "Time_Idx":    tidx_val,
            "pred":        preds_np[:, step_i],
        }))
    long_df = pd.concat(long_rows, ignore_index=True)

    del tft, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return long_df, last_tidx




# ─── TAHAP 5: FUTURE PANEL ─────────────────────────────────────────────────────
def build_future_panel(panel: pd.DataFrame, df_p: pd.DataFrame) -> pd.DataFrame:
    last_tidx = panel["Time_Idx"].max()   # 18 = Juli 2025
    last_obs  = panel[panel["Time_Idx"] == last_tidx].copy()
    mu_prop   = float(panel["Prop_Inpatient"].mean())

    # Gabungkan Tanggal_Lahir ke last_obs
    last_obs = last_obs.merge(
        df_p[["Nomor_Polis", "Tanggal_Lahir"]], on="Nomor_Polis", how="left")

    future_months = pd.date_range("2025-08-01", periods=5, freq="MS")
    h_vals = np.arange(1, 6)   # [1, 2, 3, 4, 5]

    frames = []
    for h, month in zip(h_vals, future_months):
        tmp = last_obs.copy()
        tmp["Bulan_Obs"]   = month
        tmp["Time_Idx"]    = last_tidx + h

        # Usia aktuaria dinamis
        dob = tmp["Tanggal_Lahir"]
        tmp["Usia"] = (month.year - dob.dt.year) - (
            (month.month < dob.dt.month)
            | ((month.month == dob.dt.month) & (month.day < dob.dt.day))
        ).astype(int)

        # Exponential decay → Prop_Inpatient (episodic, bukan tren permanen)
        tmp["Prop_Inpatient"] = (DECAY_ALPHA**h) * tmp["Prop_Inpatient"] + (1 - DECAY_ALPHA**h) * mu_prop

        # ICD_Group: LOCF (kategori tidak bisa di-decay)
        tmp["Total_Claim_Individu"] = 0.0
        tmp["Frekuensi_Individu"]   = 0.0
        frames.append(tmp)

    future_df = pd.concat(frames, ignore_index=True)
    future_df = future_df.drop(columns=["Tanggal_Lahir"], errors="ignore")
    for c in ["Plan_Code", "Gender", "Domisili", "ICD_Group"]:
        future_df[c] = future_df[c].astype("category")
    return future_df


# ─── CROSS-VALIDATION h=5 ──────────────────────────────────────────────────────
def run_cv(panel: pd.DataFrame) -> None:
    def mape(a, p):
        a, p = np.array(a, float), np.array(p, float)
        m = a != 0
        return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.any() else 0.0

    agg = panel.groupby("Bulan_Obs").agg(
        Freq=("Frekuensi_Individu", "sum"),
        Total=("Total_Claim_Individu", "sum"),
    )
    agg["Sev"] = agg["Total"] / agg["Freq"].replace(0, np.nan)

    print("\n[CV] h=5 rolling backtesting")
    scores = []
    for cutoff in [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]:
        train = agg[agg.index < cutoff]
        test  = agg[agg.index >= cutoff].head(5)
        if len(test) < 5:
            continue
        mu_f = train["Freq"].tail(6).mean()
        mu_t = train["Total"].tail(6).mean()
        mu_s = mu_t / max(mu_f, 1)
        sf = mape(test["Freq"], [mu_f]*5)
        st = mape(test["Total"], [mu_t]*5)
        ss = mape(test["Sev"].fillna(0), [mu_s]*5)
        avg = (sf + st + ss) / 3
        print(f"  cut={cutoff.date()} | Freq={sf:.1f}% Total={st:.1f}% Sev={ss:.1f}% → avg={avg:.1f}%")
        scores.append(avg)
    if scores:
        print(f"[CV] Mean MAPE: {np.mean(scores):.2f}%\n")


# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--klaim",  default=KLAIM_PATH)
    parser.add_argument("--polis",  default=POLIS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    panel, df_p = build_panel(args.klaim, args.polis)
    panel       = engineer_features(panel)
    print(f"[2] Panel shape: {panel.shape} | Months: 0–{panel['Time_Idx'].max()}")

    run_cv(panel)

    # Bangun full panel (historis + masa depan) untuk inferensi
    future_df  = build_future_panel(panel, df_p)
    full_panel = pd.concat([panel, future_df], ignore_index=True) \
                   .sort_values(["Nomor_Polis", "Time_Idx"]).reset_index(drop=True)
    for c in ["Plan_Code", "Gender", "Domisili", "ICD_Group"]:
        full_panel[c] = full_panel[c].astype("category")

    last_tidx  = int(panel["Time_Idx"].max())

    # Training Model 1: Total_Claim
    print("\n[4a] Total_Claim_Individu — TweedieLoss")
    ds_total = make_dataset(panel, "Total_Claim_Individu")
    df_total, _ = train_and_infer(
        ds_total, full_panel, last_tidx, TweedieLoss(p=1.5),
        "Total_Claim_Individu", "Total_Claim")

    # Training Model 2: Frekuensi
    print("\n[4b] Frekuensi_Individu — PoissonLoss")
    ds_freq = make_dataset(panel, "Frekuensi_Individu")
    df_freq, _ = train_and_infer(
        ds_freq, full_panel, last_tidx, PoissonLoss(),
        "Frekuensi_Individu", "Frekuensi")

    # Agregasi ke level bulanan
    future_tidx   = sorted(future_df["Time_Idx"].unique())
    tidx_to_month = future_df.drop_duplicates("Time_Idx") \
                              .set_index("Time_Idx")["Bulan_Obs"].to_dict()

    pt = df_total[df_total["Time_Idx"].isin(future_tidx)]
    pf = df_freq[df_freq["Time_Idx"].isin(future_tidx)]

    agg_t = pt.groupby("Time_Idx")["pred"].sum()
    agg_f = pf.groupby("Time_Idx")["pred"].sum()
    agg_s = agg_t / agg_f.replace(0, np.nan)

    rows = []
    for tidx in future_tidx:
        ms = pd.Timestamp(tidx_to_month[tidx]).strftime("%Y-%m")   # "2025-08"
        rows.extend([
            {"id": f"{ms}_Claim_Frequency", "value": round(float(agg_f.get(tidx, 0)))},
            {"id": f"{ms}_Claim_Severity",  "value": round(float(agg_s.get(tidx, 0)), 2)},
            {"id": f"{ms}_Total_Claim",     "value": round(float(agg_t.get(tidx, 0)), 2)},
        ])

    submission = pd.DataFrame(rows)
    submission.to_csv(args.output, index=False)
    print(f"\n[5] Submission saved → {args.output}")
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
