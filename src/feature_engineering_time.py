"""
Time-Domain and Thermal Health Indicator (HI) Extraction for Battery SOH Prediction.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

This module extracts 7 physically interpretable time-domain and thermal features:
  1. cc_charge_time          : Duration (s) of Constant-Current charging before cutoff (~4.2V).
  2. cv_charge_time          : Duration (s) of Constant-Voltage charging while current tapers.
  3. voltage_plateau_duration: Time elapsed (s) for discharge voltage to drop from 3.8V to 3.4V.
  4. temp_rise_discharge     : Temperature rise during discharge (max(T) - T_0) in °C.
  5. mean_temp_discharge     : Average cell temperature during discharge in °C.
  6. mean_voltage_discharge  : Time-weighted average terminal voltage during discharge in V.
  7. energy_discharge        : Integrated discharge energy in Watt-hours (Wh).
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    DATA_PROCESSED_DIR,
    RESULTS_FIG_DIR,
    RESULTS_TAB_DIR,
    NASA_CELLS,
    CALCE_CELLS,
    PRIMARY_SEED,
)

# Feature column definitions
FEATURE_COLS = [
    "cc_charge_time",
    "cv_charge_time",
    "voltage_plateau_duration",
    "temp_rise_discharge",
    "mean_temp_discharge",
    "mean_voltage_discharge",
    "energy_discharge",
]

FEATURE_LABELS = {
    "cc_charge_time": "CC Charge Time (s)",
    "cv_charge_time": "CV Charge Time (s)",
    "voltage_plateau_duration": "Voltage Plateau 3.8V→3.4V (s)",
    "temp_rise_discharge": "Discharge Temp Rise (°C)",
    "mean_temp_discharge": "Discharge Mean Temp (°C)",
    "mean_voltage_discharge": "Discharge Mean Voltage (V)",
    "energy_discharge": "Discharge Energy (Wh)",
    "soh": "State of Health (SOH %)",
}


# =====================================================================
# 1. Signal Smoothing & Outlier Filtering
# =====================================================================

def smooth_series(
    arr: np.ndarray,
    median_window: int = 5,
    savgol_window: int = 7,
    savgol_poly: int = 2,
) -> np.ndarray:
    """
    Cleans physical time-series signals by interpolating isolated NaNs,
    applying a rolling median filter to suppress impulsive single-sample noise,
    and applying a Savitzky-Golay polynomial filter for continuous gradients.
    """
    if arr is None or len(arr) == 0:
        return np.array([], dtype=np.float64)
    
    clean = np.asarray(arr, dtype=np.float64).copy()
    
    # 1. Handle NaNs/Infs via linear interpolation
    nans = np.isnan(clean) | np.isinf(clean)
    if nans.any():
        if nans.all():
            return np.zeros_like(clean)
        valid_idx = np.where(~nans)[0]
        clean[nans] = np.interp(np.where(nans)[0], valid_idx, clean[valid_idx])
    
    if len(clean) < 3:
        return clean
    
    # 2. Rolling median filter to eliminate transient sensor spikes
    w_med = min(median_window, len(clean))
    if w_med % 2 == 0:
        w_med -= 1
    if w_med >= 3:
        clean = median_filter(clean, size=w_med)
    
    # 3. Savitzky-Golay filter for smooth physical trajectories
    w_sg = min(savgol_window, len(clean))
    if w_sg % 2 == 0:
        w_sg -= 1
    if w_sg >= 5 and w_sg > savgol_poly:
        try:
            clean = savgol_filter(clean, window_length=w_sg, polyorder=savgol_poly)
        except Exception:
            pass
            
    return clean


# =====================================================================
# 2. Per-Cycle Feature Extractor
# =====================================================================

def extract_time_thermal_features(cycle_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts 7 time-domain and thermal health indicators from a single cycle dictionary.
    
    Parameters:
        cycle_dict: Dictionary containing synchronized charge & discharge profiles
                    ('ch_time', 'ch_voltage', 'ch_current', 'ch_temp',
                     'dis_time', 'dis_voltage', 'dis_current', 'dis_temp',
                     'capacity', 'soh', 'dataset', 'cell_id', 'cycle_idx')
                     
    Returns:
        Dictionary of extracted scalar features with zero NaNs guaranteed.
    """
    dataset = str(cycle_dict.get("dataset", "NASA"))
    cell_id = str(cycle_dict.get("cell_id", "Unknown"))
    cycle_idx = int(cycle_dict.get("cycle_idx", 0))
    capacity = float(cycle_dict.get("capacity", 0.0))
    soh = float(cycle_dict.get("soh", 0.0))
    
    # Clean charge time-series
    ch_time = np.asarray(cycle_dict.get("ch_time", []), dtype=np.float64)
    ch_v = smooth_series(cycle_dict.get("ch_voltage", np.array([])))
    ch_i = smooth_series(cycle_dict.get("ch_current", np.array([])))
    ch_t = smooth_series(cycle_dict.get("ch_temp", np.array([])))
    
    # Clean discharge time-series
    dis_time = np.asarray(cycle_dict.get("dis_time", []), dtype=np.float64)
    dis_v = smooth_series(cycle_dict.get("dis_voltage", np.array([])))
    dis_i = smooth_series(cycle_dict.get("dis_current", np.array([])))
    dis_t = smooth_series(cycle_dict.get("dis_temp", np.array([])))
    
    total_ch_time = float(ch_time[-1] - ch_time[0]) if len(ch_time) > 1 else 0.0
    total_dis_time = float(dis_time[-1] - dis_time[0]) if len(dis_time) > 1 else 0.0
    
    # -----------------------------------------------------------------
    # Feature 1: cc_charge_time & Feature 2: cv_charge_time
    # -----------------------------------------------------------------
    if "cc_time" in cycle_dict and "cv_time" in cycle_dict and not np.isnan(cycle_dict["cc_time"]):
        t_cc = float(cycle_dict["cc_time"])
        t_cv = float(cycle_dict["cv_time"]) if not np.isnan(cycle_dict["cv_time"]) else max(0.0, total_ch_time - t_cc)
    else:
        # Determine CC-to-CV transition from voltage and current dynamics
        if len(ch_v) > 5 and len(ch_time) > 5:
            v_max = float(np.max(ch_v))
            # Cutoff threshold: 4.195V or 99.8% of max voltage
            v_cutoff = min(4.195, v_max - 0.01) if v_max > 4.15 else (v_max * 0.99)
            idx_cc = np.where(ch_v >= v_cutoff)[0]
            
            if len(idx_cc) > 0 and idx_cc[0] > 0:
                t_cc = float(ch_time[idx_cc[0]] - ch_time[0])
                t_cv = max(0.0, total_ch_time - t_cc)
            else:
                # Fallback to current decay: current drops below 92% of peak constant charging current
                peak_i = float(np.max(ch_i)) if len(ch_i) > 0 else 1.0
                idx_decay = np.where((ch_i < 0.92 * peak_i) & (ch_time > 10.0))[0]
                if len(idx_decay) > 0:
                    t_cc = float(ch_time[idx_decay[0]] - ch_time[0])
                    t_cv = max(0.0, total_ch_time - t_cc)
                else:
                    t_cc = total_ch_time * 0.5
                    t_cv = total_ch_time * 0.5
        else:
            t_cc = total_ch_time * 0.5
            t_cv = total_ch_time * 0.5
            
    # Guarantee non-negative and finite
    t_cc = max(0.0, t_cc)
    t_cv = max(0.0, t_cv)
    if np.isnan(t_cc):
        t_cc = 0.0
    if np.isnan(t_cv):
        t_cv = max(0.0, total_ch_time - t_cc)
        
    # -----------------------------------------------------------------
    # Feature 3: voltage_plateau_duration (Time for V to drop from 3.8V to 3.4V)
    # -----------------------------------------------------------------
    if len(dis_v) > 1 and len(dis_time) > 1:
        # Sort voltage monotonically ascending for robust np.interp inversion
        sort_idx = np.argsort(dis_v)
        v_sorted = dis_v[sort_idx]
        t_sorted = dis_time[sort_idx]
        
        v_high = min(3.80, float(np.max(v_sorted)))
        v_low = max(3.40, float(np.min(v_sorted)))
        
        if v_high > v_low:
            t_high = float(np.interp(v_high, v_sorted, t_sorted))
            t_low = float(np.interp(v_low, v_sorted, t_sorted))
            voltage_plateau_duration = max(0.0, t_low - t_high)
        else:
            # Scaled proportion of total discharge duration if span does not encompass 3.8-3.4V
            voltage_plateau_duration = total_dis_time * 0.45
    else:
        voltage_plateau_duration = 0.0
        
    if np.isnan(voltage_plateau_duration) or np.isinf(voltage_plateau_duration):
        voltage_plateau_duration = 0.0
        
    # -----------------------------------------------------------------
    # Feature 4: temp_rise_discharge (max(T) - T_0)
    # -----------------------------------------------------------------
    if len(dis_t) > 0:
        temp_rise_discharge = float(np.max(dis_t) - dis_t[0])
    else:
        temp_rise_discharge = 0.0
    if np.isnan(temp_rise_discharge) or np.isinf(temp_rise_discharge):
        temp_rise_discharge = 0.0
        
    # -----------------------------------------------------------------
    # Feature 5: mean_temp_discharge (Average temperature during discharge)
    # -----------------------------------------------------------------
    if len(dis_t) > 0:
        mean_temp_discharge = float(np.mean(dis_t))
    else:
        mean_temp_discharge = 25.0
    if np.isnan(mean_temp_discharge) or np.isinf(mean_temp_discharge):
        mean_temp_discharge = 25.0
        
    # -----------------------------------------------------------------
    # Feature 6: mean_voltage_discharge (Time-weighted average terminal voltage)
    # -----------------------------------------------------------------
    if len(dis_v) > 1 and len(dis_time) > 1 and total_dis_time > 0:
        dt = np.diff(dis_time)
        v_mid = 0.5 * (dis_v[:-1] + dis_v[1:])
        mean_voltage_discharge = float(np.sum(v_mid * dt) / total_dis_time)
    elif len(dis_v) > 0:
        mean_voltage_discharge = float(np.mean(dis_v))
    else:
        mean_voltage_discharge = 3.65
    if np.isnan(mean_voltage_discharge) or np.isinf(mean_voltage_discharge):
        mean_voltage_discharge = 3.65
        
    # -----------------------------------------------------------------
    # Feature 7: energy_discharge (Integrated discharge energy in Wh)
    # -----------------------------------------------------------------
    if len(dis_v) > 1 and len(dis_i) > 1 and len(dis_time) > 1 and total_dis_time > 0:
        p = dis_v * np.abs(dis_i)
        dt = np.diff(dis_time)
        p_mid = 0.5 * (p[:-1] + p[1:])
        energy_discharge = float(np.sum(p_mid * dt) / 3600.0)
    elif len(dis_v) > 0 and len(dis_i) > 0:
        energy_discharge = float(np.mean(dis_v) * np.mean(np.abs(dis_i)) * total_dis_time / 3600.0)
    else:
        energy_discharge = float(capacity * 3.7)
    if np.isnan(energy_discharge) or np.isinf(energy_discharge):
        energy_discharge = float(capacity * 3.7)

    return {
        "dataset": dataset,
        "cell_id": cell_id,
        "cycle_idx": cycle_idx,
        "capacity": capacity,
        "soh": soh,
        "cc_charge_time": t_cc,
        "cv_charge_time": t_cv,
        "voltage_plateau_duration": voltage_plateau_duration,
        "temp_rise_discharge": temp_rise_discharge,
        "mean_temp_discharge": mean_temp_discharge,
        "mean_voltage_discharge": mean_voltage_discharge,
        "energy_discharge": energy_discharge,
    }


# =====================================================================
# 3. Multi-Cell Extraction & Dataset Compilation
# =====================================================================

def extract_all_features(
    parsed_cycles_input: Optional[Union[Dict[str, List[Dict[str, Any]]], str, Path]] = None,
    save_output: bool = True,
    csv_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Iterates over all battery cells and cycles, extracting the 7 Health Indicators.
    Imputes any remaining NaNs using cycle-wise linear interpolation, and saves
    the resulting DataFrame to disk.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path or (DATA_PROCESSED_DIR / "time_thermal_features.csv"))
    
    # Load parsed cycles if not passed in memory
    if parsed_cycles_input is None or isinstance(parsed_cycles_input, (str, Path)):
        pkl_path = Path(parsed_cycles_input or (DATA_PROCESSED_DIR / "parsed_cycles.pkl"))
        if not pkl_path.exists():
            from src.data_loader import load_all_battery_datasets
            if verbose:
                print(f"Parsed cycles not found at {pkl_path}. Loading and parsing raw datasets...")
            all_cycles = load_all_battery_datasets(save_processed=True, verbose=verbose)
        else:
            if verbose:
                print(f"Loading parsed cycles from {pkl_path.name}...")
            with open(pkl_path, "rb") as f:
                all_cycles = pickle.load(f)
    else:
        all_cycles = parsed_cycles_input
        
    records: List[Dict[str, Any]] = []
    
    if verbose:
        print("\nExtracting Time-Domain & Thermal Health Indicators:")
        
    for cell_id, cycles in all_cycles.items():
        if verbose:
            print(f"  * Processing {cell_id:<8} ({len(cycles)} cycles)...", end="", flush=True)
        cell_records = [extract_time_thermal_features(c) for c in cycles]
        records.extend(cell_records)
        if verbose:
            print(f" Done ({len(cell_records)} feature vectors)")
            
    df = pd.DataFrame(records)
    
    # -----------------------------------------------------------------
    # Zero NaN Guarantee: per-cell interpolation and boundary filling
    # -----------------------------------------------------------------
    for cell_id in df["cell_id"].unique():
        mask = df["cell_id"] == cell_id
        for col in FEATURE_COLS:
            if df.loc[mask, col].isna().any():
                df.loc[mask, col] = (
                    df.loc[mask, col]
                    .interpolate(method="linear", limit_direction="both")
                    .bfill()
                    .ffill()
                    .fillna(0.0)
                )
                
    # Final global safety check
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    assert df.isna().sum().sum() == 0, f"Encountered unexpected NaNs: {df.isna().sum().to_dict()}"
    
    if save_output:
        df.to_csv(csv_path, index=False)
        parquet_path = csv_path.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
        if verbose:
            print(f"\n[OK] Saved intermediate feature table to {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")
            
    return df


# =====================================================================
# 4. Summary Statistics & Correlation Analysis
# =====================================================================

def compute_correlation_matrices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Computes Pearson and Spearman correlation matrices for overall, NASA, and CALCE subsets.
    """
    cols = FEATURE_COLS + ["soh"]
    
    corr_dict = {
        "overall_pearson": df[cols].corr(method="pearson"),
        "overall_spearman": df[cols].corr(method="spearman"),
    }
    
    df_nasa = df[df["dataset"] == "NASA"]
    if len(df_nasa) > 0:
        corr_dict["nasa_pearson"] = df_nasa[cols].corr(method="pearson")
        corr_dict["nasa_spearman"] = df_nasa[cols].corr(method="spearman")
        
    df_calce = df[df["dataset"] == "CALCE"]
    if len(df_calce) > 0:
        # In CALCE, thermal chamber was strictly constant 25.0°C (zero variance)
        # Drop constant thermal columns for CALCE correlation matrix
        calce_cols = [c for c in cols if df_calce[c].std() > 1e-6]
        corr_dict["calce_pearson"] = df_calce[calce_cols].corr(method="pearson")
        corr_dict["calce_spearman"] = df_calce[calce_cols].corr(method="spearman")
        
    return corr_dict


def generate_feature_summary_table(df: pd.DataFrame, print_table: bool = True) -> pd.DataFrame:
    """
    Generates a statistical summary table of extracted features with correlation against SOH.
    """
    RESULTS_TAB_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    
    for feat in FEATURE_COLS:
        all_r = df[feat].corr(df["soh"])
        df_nasa = df[df["dataset"] == "NASA"]
        nasa_r = df_nasa[feat].corr(df_nasa["soh"]) if len(df_nasa) > 0 else np.nan
        df_calce = df[df["dataset"] == "CALCE"]
        calce_r = df_calce[feat].corr(df_calce["soh"]) if (len(df_calce) > 0 and df_calce[feat].std() > 1e-6) else np.nan
        
        summary_rows.append({
            "Feature": feat,
            "Physical Unit": FEATURE_LABELS[feat].split("(")[-1].replace(")", "") if "(" in FEATURE_LABELS[feat] else "-",
            "Min": round(float(df[feat].min()), 2),
            "Max": round(float(df[feat].max()), 2),
            "Mean": round(float(df[feat].mean()), 2),
            "Std": round(float(df[feat].std()), 2),
            "Overall r": round(float(all_r), 4),
            "NASA r": round(float(nasa_r), 4) if not np.isnan(nasa_r) else "N/A",
            "CALCE r": round(float(calce_r), 4) if not np.isnan(calce_r) else "Const (25°C)",
        })
        
    df_summary = pd.DataFrame(summary_rows)
    tab_path = RESULTS_TAB_DIR / "time_thermal_features_summary.csv"
    df_summary.to_csv(tab_path, index=False)
    
    if print_table:
        print("\n" + "=" * 108)
        print(" WD-GPR-SR Time-Domain & Thermal Health Indicators: Statistical & Correlation Summary")
        print("=" * 108)
        print(f"{'Feature':<26} {'Unit':<6} {'Min':<9} {'Max':<9} {'Mean':<9} {'Std':<9} {'All r':<9} {'NASA r':<9} {'CALCE r':<12}")
        print("-" * 108)
        for _, r in df_summary.iterrows():
            nasa_str = f"{r['NASA r']:+.4f}" if isinstance(r['NASA r'], float) else str(r['NASA r'])
            calce_str = f"{r['CALCE r']:+.4f}" if isinstance(r['CALCE r'], float) else str(r['CALCE r'])
            print(f"{r['Feature']:<26} {r['Physical Unit']:<6} {r['Min']:<9.2f} {r['Max']:<9.2f} {r['Mean']:<9.2f} {r['Std']:<9.2f} {r['Overall r']:<+9.4f} {nasa_str:<9} {calce_str:<12}")
        print("=" * 108)
        
    return df_summary


# =====================================================================
# 5. Visualization: Correlation Heatmaps & Feature Trends
# =====================================================================

def plot_feature_correlations(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots multi-panel publication-grade correlation heatmaps:
      - Panel (a): NASA PCoE Pearson correlation heatmap.
      - Panel (b): CALCE (UMD) Pearson correlation heatmap.
      - Panel (c): Overall dataset correlation heatmap.
      - Panel (d): Feature-to-SOH Pearson vs. Spearman correlation comparison.
    """
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path or (RESULTS_FIG_DIR / "time_thermal_correlation_heatmap.png"))
    
    sns.set_theme(style="white", font_scale=0.9)
    fig = plt.figure(figsize=(16, 14), dpi=300)
    
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    
    # Readable short labels
    short_labels = {
        "cc_charge_time": "CC Charge Time",
        "cv_charge_time": "CV Charge Time",
        "voltage_plateau_duration": "Plateau Duration",
        "temp_rise_discharge": "Temp Rise (ΔT)",
        "mean_temp_discharge": "Mean Temp",
        "mean_voltage_discharge": "Mean Voltage",
        "energy_discharge": "Energy (Wh)",
        "soh": "SOH (%)",
    }
    
    cols = FEATURE_COLS + ["soh"]
    display_names = [short_labels[c] for c in cols]
    
    # -----------------------------------------------------------------
    # Panel (a): NASA Pearson Heatmap
    # -----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    df_nasa = df[df["dataset"] == "NASA"]
    corr_nasa = df_nasa[cols].corr(method="pearson")
    corr_nasa.columns = display_names
    corr_nasa.index = display_names
    
    mask_nasa = np.triu(np.ones_like(corr_nasa, dtype=bool), k=1)
    sns.heatmap(
        corr_nasa,
        mask=mask_nasa,
        annot=True,
        fmt="+.2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax_a,
    )
    ax_a.set_title("(a) NASA PCoE: Ambient Thermal Cycling Correlation Matrix", fontweight="bold", pad=12)
    ax_a.tick_params(axis="x", rotation=45)
    
    # -----------------------------------------------------------------
    # Panel (b): CALCE Pearson Heatmap
    # -----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    df_calce = df[df["dataset"] == "CALCE"]
    calce_cols = [c for c in cols if df_calce[c].std() > 1e-6]
    calce_display = [short_labels[c] for c in calce_cols]
    corr_calce = df_calce[calce_cols].corr(method="pearson")
    corr_calce.columns = calce_display
    corr_calce.index = calce_display
    
    mask_calce = np.triu(np.ones_like(corr_calce, dtype=bool), k=1)
    sns.heatmap(
        corr_calce,
        mask=mask_calce,
        annot=True,
        fmt="+.2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax_b,
    )
    ax_b.set_title("(b) CALCE (UMD): Isothermal Chamber (25°C) Correlation Matrix", fontweight="bold", pad=12)
    ax_b.tick_params(axis="x", rotation=45)
    
    # -----------------------------------------------------------------
    # Panel (c): Overall Combined Heatmap
    # -----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    corr_all = df[cols].corr(method="pearson")
    corr_all.columns = display_names
    corr_all.index = display_names
    
    mask_all = np.triu(np.ones_like(corr_all, dtype=bool), k=1)
    sns.heatmap(
        corr_all,
        mask=mask_all,
        annot=True,
        fmt="+.2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax_c,
    )
    ax_c.set_title("(c) Overall Benchmark: Combined Correlation Matrix", fontweight="bold", pad=12)
    ax_c.tick_params(axis="x", rotation=45)
    
    # -----------------------------------------------------------------
    # Panel (d): Feature-to-SOH Correlation Bar Chart
    # -----------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    
    pearson_soh = [df_nasa[c].corr(df_nasa["soh"]) for c in FEATURE_COLS]
    spearman_soh = [df_nasa[c].corr(df_nasa["soh"], method="spearman") for c in FEATURE_COLS]
    short_feature_names = [short_labels[c] for c in FEATURE_COLS]
    
    y_pos = np.arange(len(short_feature_names))
    height = 0.35
    
    ax_d.barh(y_pos - height/2, pearson_soh, height, label="Pearson r (Linear)", color="#2b5c8f", alpha=0.9)
    ax_d.barh(y_pos + height/2, spearman_soh, height, label="Spearman ρ (Monotonic)", color="#d95f02", alpha=0.9)
    
    ax_d.axvline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ax_d.axvline(0.8, color="green", linestyle="--", linewidth=1.0, alpha=0.6, label="Strong Corr Threshold (±0.8)")
    ax_d.axvline(-0.8, color="green", linestyle="--", linewidth=1.0, alpha=0.6)
    
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(short_feature_names)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Correlation with SOH")
    ax_d.set_title("(d) Health Indicator Sensitivity to SOH (NASA Subset)", fontweight="bold", pad=12)
    ax_d.set_xlim(-1.05, 1.05)
    ax_d.legend(loc="lower left", frameon=True)
    ax_d.grid(True, linestyle="--", alpha=0.5, axis="x")
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved correlation heatmaps to {save_path}")


def plot_feature_vs_soh_trends(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots individual Health Indicator vs. SOH scatter plots colored by battery cell.
    """
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path or (RESULTS_FIG_DIR / "feature_vs_soh_trends.png"))
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
    axes = axes.flatten()
    
    cell_palette = {
        "B0005": "#1f77b4", "B0006": "#ff7f0e", "B0007": "#2ca02c", "B0018": "#d62728",
        "CS2_35": "#9467bd", "CS2_36": "#8c564b", "CS2_37": "#e377c2", "CS2_38": "#7f7f7f"
    }
    
    for idx, feat in enumerate(FEATURE_COLS):
        ax = axes[idx]
        for cell_id in df["cell_id"].unique():
            cell_data = df[df["cell_id"] == cell_id]
            ax.scatter(
                cell_data[feat],
                cell_data["soh"],
                label=cell_id,
                color=cell_palette.get(cell_id, "#333333"),
                alpha=0.6,
                s=12,
            )
            
        ax.set_title(FEATURE_LABELS[feat], fontweight="bold", fontsize=10)
        ax.set_xlabel(FEATURE_LABELS[feat].split("(")[-1].replace(")", "") if "(" in FEATURE_LABELS[feat] else feat)
        ax.set_ylabel("SOH (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    # Legend in the 8th subplot
    axes[7].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[7].legend(handles, labels, loc="center", title="Battery Cell ID", frameon=True, fontsize=10, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved feature vs. SOH trend figures to {save_path}")


# =====================================================================
# Main Execution Runner
# =====================================================================

if __name__ == "__main__":
    df_features = extract_all_features(save_output=True, verbose=True)
    summary_df = generate_feature_summary_table(df_features, print_table=True)
    plot_feature_correlations(df_features, show=False)
    plot_feature_vs_soh_trends(df_features, show=False)

