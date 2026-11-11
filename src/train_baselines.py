"""
Comprehensive Baseline Training and Evaluation Harness across LOCO Folds.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

Executes the 13 competitive baseline models across 8 LOCO folds and 3 seeds [42, 100, 2026]:
  1. Classical & Tree-Based : SVR, Random Forest, XGBoost
  2. Symbolic Regression    : Standalone Symbolic Regressor (PySR / gplearn)
  3. Gaussian Processes     : Exact GPR (RBF, Matérn, ARD), Sparse Variational GPR, Deep-Kernel GPR
  4. Deep & Recurrent NNs   : WideNN, DeepNN, Wide & Deep NN, BiLSTM (Sequential)
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Ensure project root is in sys.path and suppress Loky CPU probe & Matplotlib warnings
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    SEEDS,
    PRIMARY_SEED,
    DATA_PROCESSED_DIR,
    RESULTS_FIG_DIR,
    RESULTS_TAB_DIR,
    NASA_CELLS,
    CALCE_CELLS,
    set_global_seed,
)

from src.baselines import (
    train_svr_baseline,
    train_rf_baseline,
    train_xgboost_baseline,
    train_standalone_sr_baseline,
    train_gp_model,
    train_pytorch_nn,
    train_bilstm_baseline,
    WideNN,
    DeepNN,
    WideAndDeepNN,
)

BASELINE_MODELS = [
    "SVR",
    "Random Forest",
    "XGBoost",
    "Standalone SR",
    "Exact GPR (RBF)",
    "Exact GPR (Matérn)",
    "Exact GPR (ARD)",
    "Sparse Variational GPR",
    "Deep-Kernel GPR",
    "WideNN",
    "DeepNN",
    "Wide & Deep NN",
    "BiLSTM",
]

MODEL_FAMILIES = {
    "SVR": "Classical",
    "Random Forest": "Tree-Based",
    "XGBoost": "Tree-Based",
    "Standalone SR": "Symbolic Regression",
    "Exact GPR (RBF)": "Gaussian Process",
    "Exact GPR (Matérn)": "Gaussian Process",
    "Exact GPR (ARD)": "Gaussian Process",
    "Sparse Variational GPR": "Gaussian Process",
    "Deep-Kernel GPR": "Gaussian Process",
    "WideNN": "Neural Network",
    "DeepNN": "Neural Network",
    "Wide & Deep NN": "Neural Network",
    "BiLSTM": "Sequence Model",
}


# =====================================================================
# 1. Single Fold-Seed Model Dispatcher
# =====================================================================

def run_model_on_fold(
    model_name: str,
    fold_data: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a single baseline model on a pre-split LOCO fold under a given random seed.
    """
    set_global_seed(seed)
    
    X_tr = fold_data["X_train"]
    y_tr = fold_data["y_train"]
    X_te = fold_data["X_test"]
    y_te = fold_data["y_test"]
    
    X_tr_seq = fold_data["X_train_seq"]
    X_te_seq = fold_data["X_test_seq"]
    
    input_dim = X_tr.shape[1]
    is_large = len(X_tr) > 1000
    
    if model_name == "SVR":
        return train_svr_baseline(X_tr, y_tr, X_te, y_te, seed=seed)
        
    elif model_name == "Random Forest":
        return train_rf_baseline(X_tr, y_tr, X_te, y_te, seed=seed, n_estimators=100)
        
    elif model_name == "XGBoost":
        return train_xgboost_baseline(X_tr, y_tr, X_te, y_te, seed=seed, n_estimators=100, learning_rate=0.1)
        
    elif model_name == "Standalone SR":
        gen = 8 if is_large else 12
        pop = 200 if is_large else 350
        return train_standalone_sr_baseline(X_tr, y_tr, X_te, y_te, seed=seed, generations=gen, population_size=pop)
        
    elif model_name == "Exact GPR (RBF)":
        epochs = 20 if is_large else 35
        return train_gp_model(X_tr, y_tr, X_te, y_te, gp_type="rbf", seed=seed, num_epochs=epochs)
        
    elif model_name == "Exact GPR (Matérn)":
        epochs = 20 if is_large else 35
        return train_gp_model(X_tr, y_tr, X_te, y_te, gp_type="matern", seed=seed, num_epochs=epochs)
        
    elif model_name == "Exact GPR (ARD)":
        epochs = 20 if is_large else 35
        return train_gp_model(X_tr, y_tr, X_te, y_te, gp_type="ard", seed=seed, num_epochs=epochs)
        
    elif model_name == "Sparse Variational GPR":
        epochs = 20 if is_large else 30
        m_pts = 64 if is_large else 32
        return train_gp_model(X_tr, y_tr, X_te, y_te, gp_type="sparse", seed=seed, num_inducing=m_pts, num_epochs=epochs)
        
    elif model_name == "Deep-Kernel GPR":
        epochs = 20 if is_large else 30
        return train_gp_model(X_tr, y_tr, X_te, y_te, gp_type="dkl", seed=seed, num_epochs=epochs)
        
    elif model_name == "WideNN":
        epochs = 30 if is_large else 40
        batch_size = 128 if is_large else 64
        return train_pytorch_nn(WideNN(input_dim), X_tr, y_tr, X_te, y_te, "WideNN", seed=seed, epochs=epochs, batch_size=batch_size)
        
    elif model_name == "DeepNN":
        epochs = 30 if is_large else 40
        batch_size = 128 if is_large else 64
        return train_pytorch_nn(DeepNN(input_dim), X_tr, y_tr, X_te, y_te, "DeepNN", seed=seed, epochs=epochs, batch_size=batch_size)
        
    elif model_name == "Wide & Deep NN":
        epochs = 30 if is_large else 40
        batch_size = 128 if is_large else 64
        return train_pytorch_nn(WideAndDeepNN(input_dim), X_tr, y_tr, X_te, y_te, "Wide & Deep NN", seed=seed, epochs=epochs, batch_size=batch_size)
        
    elif model_name == "BiLSTM":
        epochs = 25 if is_large else 30
        batch_size = 128 if is_large else 64
        return train_bilstm_baseline(X_tr_seq, y_tr, X_te_seq, y_te, seed=seed, epochs=epochs, batch_size=batch_size)
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# =====================================================================
# 2. End-to-End Evaluation Runner across Datasets & Seeds
# =====================================================================

def evaluate_all_baselines(
    splits_dict: Optional[Dict[str, Any]] = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    save_raw: bool = True,
    save_agg: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes all baseline models across NASA and CALCE benchmarks under 3 seeds [42, 100, 2026].
    Records point accuracy metrics (MAE, MSE, RMSE, R2, MAPE) and calibrated uncertainty metrics (NLL, PICP, MPIW).
    Features incremental saving and automatic resumption from existing CSV records.
    """
    RESULTS_TAB_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if splits_dict is None:
        splits_path = DATA_PROCESSED_DIR / "loco_splits.pkl"
        with open(splits_path, "rb") as f:
            splits_dict = pickle.load(f)
            
    datasets = datasets or ["NASA", "CALCE"]
    seeds = seeds or SEEDS
    models = models or BASELINE_MODELS
    
    raw_csv_path = RESULTS_TAB_DIR / "baselines_raw_runs.csv"
    existing_runs = set()
    raw_records: List[Dict[str, Any]] = []
    
    if raw_csv_path.exists():
        try:
            df_existing = pd.read_csv(raw_csv_path)
            raw_records = df_existing.to_dict("records")
            for r in raw_records:
                existing_runs.add((r["dataset"], str(r["test_cell"]), int(r["seed"]), r["model"]))
            if verbose:
                print(f"[INFO] Loaded {len(existing_runs)} pre-existing run records from {raw_csv_path}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not parse existing {raw_csv_path}: {e}", flush=True)
                
    total_runs = len(datasets) * 4 * len(seeds) * len(models)
    run_counter = 0
    t_start = time.time()
    
    if verbose:
        print("=" * 105, flush=True)
        print(f" WD-GPR-SR Baseline Benchmark Evaluation ({len(models)} models x {len(datasets)} datasets x 4 folds x {len(seeds)} seeds = {total_runs} runs)", flush=True)
        print("=" * 105, flush=True)
        
    for ds in datasets:
        ds_folds = splits_dict[ds]
        for test_cell, fold_data in ds_folds.items():
            for seed in seeds:
                for model_name in models:
                    run_counter += 1
                    key = (ds, str(test_cell), int(seed), model_name)
                    if key in existing_runs:
                        continue
                        
                    t0 = time.time()
                    try:
                        res = run_model_on_fold(model_name, fold_data, seed=seed)
                        m = res["metrics"]
                        elapsed = time.time() - t0
                        
                        rec = {
                            "dataset": ds,
                            "test_cell": test_cell,
                            "fold_idx": fold_data["fold_idx"],
                            "model": model_name,
                            "family": MODEL_FAMILIES.get(model_name, "Other"),
                            "seed": seed,
                            "MAE": m.get("MAE", np.nan),
                            "MSE": m.get("MSE", np.nan),
                            "RMSE": m.get("RMSE", np.nan),
                            "R2": m.get("R2", np.nan),
                            "MAPE": m.get("MAPE", np.nan),
                            "NLL": m.get("NLL", np.nan),
                            "PICP_95": m.get("PICP_95", np.nan),
                            "MPIW_95": m.get("MPIW_95", np.nan),
                            "elapsed_s": round(elapsed, 2),
                        }
                        raw_records.append(rec)
                        existing_runs.add(key)
                        
                        # Incremental write to prevent data loss
                        if save_raw:
                            df_temp = pd.DataFrame(raw_records)
                            df_temp.to_csv(raw_csv_path, index=False)
                            
                        if verbose:
                            nll_str = f" | NLL: {m['NLL']:.2f}" if "NLL" in m and not np.isnan(m["NLL"]) else ""
                            print(f"  [{run_counter:3d}/{total_runs}] {ds:<5} | Test: {test_cell:<8} | Seed: {seed:<4} | {model_name:<22} | RMSE: {m['RMSE']:.4f} | R²: {m['R2']:.4f}{nll_str} ({elapsed:.1f}s)", flush=True)
                            
                    except Exception as e:
                        print(f"  [ERROR] {ds} {test_cell} {model_name} (seed {seed}): {e}", flush=True)
                        
    df_raw = pd.DataFrame(raw_records)
    if save_raw:
        df_raw.to_csv(raw_csv_path, index=False)
        if verbose:
            print(f"\n[OK] Saved all {len(df_raw)} raw baseline run records to {raw_csv_path}", flush=True)
            
    # -----------------------------------------------------------------
    # Compute Aggregated mean ± std Table Across 3 Seeds
    # -----------------------------------------------------------------
    metric_cols = ["RMSE", "MAE", "R2", "MAPE", "NLL", "PICP_95", "MPIW_95"]
    
    agg_rows = []
    for ds in datasets:
        df_ds = df_raw[df_raw["dataset"] == ds]
        for model_name in models:
            sub = df_ds[df_ds["model"] == model_name]
            if len(sub) == 0:
                continue
                
            row = {
                "Dataset": ds,
                "Model": model_name,
                "Family": MODEL_FAMILIES.get(model_name, "Other"),
                "Total Runs": len(sub),
            }
            
            for m_col in metric_cols:
                vals = sub[m_col].dropna()
                if len(vals) > 0:
                    mu = vals.mean()
                    sd = vals.std()
                    # Format as mean ± std
                    if m_col in ["RMSE", "MAE", "MAPE"]:
                        row[m_col] = f"{mu:.3f} ± {sd:.3f}"
                    elif m_col == "R2":
                        row[m_col] = f"{mu:.4f} ± {sd:.4f}"
                    elif m_col == "NLL":
                        row[m_col] = f"{mu:.2f} ± {sd:.2f}"
                    elif m_col in ["PICP_95", "MPIW_95"]:
                        row[m_col] = f"{mu:.1f} ± {sd:.1f}"
                    row[f"{m_col}_mean"] = mu
                    row[f"{m_col}_std"] = sd
                else:
                    row[m_col] = "—"
                    row[f"{m_col}_mean"] = np.nan
                    row[f"{m_col}_std"] = np.nan
                    
            agg_rows.append(row)
            
    df_agg = pd.DataFrame(agg_rows)
    
    if save_agg:
        agg_csv_path = RESULTS_TAB_DIR / "baselines_comparison.csv"
        df_agg.to_csv(agg_csv_path, index=False)
        if verbose:
            print(f"[OK] Saved aggregated baseline comparison table to {agg_csv_path}")
            
    total_time = time.time() - t_start
    if verbose:
        print(f"\nExecution finished in {total_time/60:.2f} minutes.")
        print_comparison_table(df_agg)
        
    return df_raw, df_agg


# =====================================================================
# 3. Formatted Table & Publication Plotting
# =====================================================================

def print_comparison_table(df_agg: pd.DataFrame):
    """Prints clean terminal summary table of baseline model comparisons."""
    print("\n" + "=" * 128)
    print(" WD-GPR-SR Baseline Benchmark Evaluation Summary (Mean ± Std across 4 LOCO Folds x 3 Seeds)")
    print("=" * 128)
    print(f"{'Dataset':<8} {'Model':<24} {'Family':<18} {'RMSE (%)':<16} {'MAE (%)':<16} {'R²':<18} {'NLL':<14} {'PICP (95%)':<14}")
    print("-" * 128)
    for _, r in df_agg.iterrows():
        print(f"{r['Dataset']:<8} {r['Model']:<24} {r['Family']:<18} {r['RMSE']:<16} {r['MAE']:<16} {r['R2']:<18} {r['NLL']:<14} {r['PICP_95']:<14}")
    print("=" * 128)


def plot_baseline_comparison(
    df_agg: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots multi-panel comparison bar charts for RMSE, MAE, R², and NLL across baseline models.
    """
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path or (RESULTS_FIG_DIR / "baselines_benchmark_comparison.png"))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # 1. NASA RMSE Comparison
    ax = axes[0, 0]
    sub_nasa = df_agg[df_agg["Dataset"] == "NASA"].sort_values("RMSE_mean")
    ax.barh(sub_nasa["Model"], sub_nasa["RMSE_mean"], xerr=sub_nasa["RMSE_std"], color="#2b5c8f", alpha=0.85, capsize=4)
    ax.set_title("(a) NASA PCoE Benchmark: RMSE (SOH %)", fontweight="bold")
    ax.set_xlabel("Root Mean Squared Error (%)")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.invert_yaxis()
    
    # 2. CALCE RMSE Comparison
    ax = axes[0, 1]
    sub_calce = df_agg[df_agg["Dataset"] == "CALCE"].sort_values("RMSE_mean")
    ax.barh(sub_calce["Model"], sub_calce["RMSE_mean"], xerr=sub_calce["RMSE_std"], color="#d95f02", alpha=0.85, capsize=4)
    ax.set_title("(b) CALCE (UMD) Benchmark: RMSE (SOH %)", fontweight="bold")
    ax.set_xlabel("Root Mean Squared Error (%)")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.invert_yaxis()
    
    # 3. Overall R² Comparison
    ax = axes[1, 0]
    models = df_agg["Model"].unique()
    y_pos = np.arange(len(models))
    h = 0.38
    
    def get_val(df, ds, m, col):
        v = df[(df["Dataset"] == ds) & (df["Model"] == m)][col].values
        return v[0] if len(v) > 0 else np.nan

    nasa_r2 = [get_val(df_agg, "NASA", m, "R2_mean") for m in models]
    calce_r2 = [get_val(df_agg, "CALCE", m, "R2_mean") for m in models]
    
    ax.barh(y_pos - h/2, np.clip(np.nan_to_num(nasa_r2, nan=-0.2), -0.2, 1.0), h, label="NASA PCoE", color="#2b5c8f", alpha=0.9)
    ax.barh(y_pos + h/2, np.clip(np.nan_to_num(calce_r2, nan=-0.2), -0.2, 1.0), h, label="CALCE (UMD)", color="#d95f02", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("(c) Generalization Coefficient of Determination (R²)", fontweight="bold")
    ax.set_xlabel("R² Score (Clipped to [-0.2, 1.0])")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.invert_yaxis()
    
    # 4. Uncertainty Calibration: NLL & PICP for Probabilistic Models (GPRs)
    ax = axes[1, 1]
    gp_sub = df_agg[df_agg["Family"] == "Gaussian Process"].dropna(subset=["NLL_mean"])
    gp_models = gp_sub["Model"].unique()
    y_gp = np.arange(len(gp_models))
    
    nasa_nll = [get_val(df_agg, "NASA", m, "NLL_mean") for m in gp_models]
    calce_nll = [get_val(df_agg, "CALCE", m, "NLL_mean") for m in gp_models]
    
    ax.barh(y_gp - h/2, nasa_nll, h, label="NASA PCoE", color="#2b5c8f", alpha=0.9)
    ax.barh(y_gp + h/2, calce_nll, h, label="CALCE (UMD)", color="#d95f02", alpha=0.9)
    ax.set_yticks(y_gp)
    ax.set_yticklabels(gp_models, fontsize=9)
    ax.set_title("(d) Uncertainty Quality: Negative Log-Likelihood (NLL)", fontweight="bold")
    ax.set_xlabel("Gaussian NLL (Lower is Better)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved baseline comparison figure to {save_path}")


# =====================================================================
# Main Execution Runner
# =====================================================================

if __name__ == "__main__":
    df_raw, df_agg = evaluate_all_baselines(save_raw=True, save_agg=True, verbose=True)
    plot_baseline_comparison(df_agg, show=False)

