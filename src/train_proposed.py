"""
Training and Evaluation Pipeline for Proposed WD-GPR-SR Model across LOCO Folds.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

Executes the proposed Wide-and-Deep Sparse GPR with Uncertainty-Weighted Symbolic
Residual correction across all 8 Leave-One-Cell-Out (LOCO) folds and 3 seeds [42, 100, 2026].
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Ensure project root is in sys.path
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
    set_global_seed,
)
from src.metrics import evaluate_predictions, count_sympy_nodes
from src.proposed_model import WD_GPR_SR_Pipeline


# =====================================================================
# 1. Single Fold-Seed Evaluation Dispatcher
# =====================================================================

def run_proposed_on_fold(
    fold_data: Dict[str, Any],
    seed: int = 42,
    epochs_gpr: int = 35,
    generations_sr: int = 10,
    population_size_sr: int = 200,
    num_inducing: int = 64,
) -> Dict[str, Any]:
    """
    Trains and evaluates the proposed WD-GPR-SR pipeline on a single fold and seed.
    """
    set_global_seed(seed)
    
    X_tr = fold_data["X_train"]
    y_tr = fold_data["y_train"]
    X_te = fold_data["X_test"]
    y_te = fold_data["y_test"]
    feature_names = fold_data.get("feature_names", [f"HI_{i+1}" for i in range(X_tr.shape[1])])
    
    is_large = len(X_tr) > 1000
    ep_gpr = 25 if is_large else epochs_gpr
    gen_sr = 8 if is_large else generations_sr
    pop_sr = 150 if is_large else population_size_sr
    m_pts = 64 if is_large else min(num_inducing, len(X_tr))
    
    pipeline = WD_GPR_SR_Pipeline(
        input_dim=X_tr.shape[1],
        deep_dim=8,
        num_inducing=m_pts,
        epochs_gpr=ep_gpr,
        batch_size=128 if is_large else 64,
        lr_gpr=0.03,
        generations_sr=gen_sr,
        population_size_sr=pop_sr,
        random_state=seed,
    )
    
    t0 = time.time()
    pipeline.fit(X_tr, y_tr, feature_names=feature_names, verbose=False)
    fit_time = time.time() - t0
    
    # Predict on test set
    t1 = time.time()
    y_pred, y_var, mu_wd, res_sr = pipeline.predict(X_te, return_decomposed=True)
    infer_time = time.time() - t1
    
    best_equation = pipeline.get_best_equation()
    metrics = evaluate_predictions(y_te, y_pred, y_var=y_var, expr=best_equation)
    
    # Also record WD-GPR baseline metrics (without SR residual correction) for ablation
    metrics_wd_alone = evaluate_predictions(y_te, mu_wd, y_var=y_var)
    
    return {
        "metrics": metrics,
        "metrics_wd_alone": metrics_wd_alone,
        "y_pred": y_pred,
        "y_var": y_var,
        "mu_wd": mu_wd,
        "res_sr": res_sr,
        "equation": best_equation,
        "fit_time": fit_time,
        "infer_time": infer_time,
    }


# =====================================================================
# 2. Comprehensive Runner across Datasets & Seeds
# =====================================================================

def evaluate_proposed_model(
    splits_dict: Optional[Dict[str, Any]] = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    save_raw: bool = True,
    append_to_baselines: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the proposed WD-GPR-SR across all NASA and CALCE folds for 3 seeds [42, 100, 2026].
    """
    RESULTS_TAB_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if splits_dict is None:
        splits_path = DATA_PROCESSED_DIR / "loco_splits.pkl"
        with open(splits_path, "rb") as f:
            splits_dict = pickle.load(f)
            
    datasets = datasets or ["NASA", "CALCE"]
    seeds = seeds or SEEDS
    
    raw_csv_path = RESULTS_TAB_DIR / "proposed_raw_runs.csv"
    existing_runs = set()
    raw_records: List[Dict[str, Any]] = []
    
    if raw_csv_path.exists():
        try:
            df_existing = pd.read_csv(raw_csv_path)
            raw_records = df_existing.to_dict("records")
            for r in raw_records:
                existing_runs.add((r["dataset"], str(r["test_cell"]), int(r["seed"])))
            if verbose:
                print(f"[INFO] Loaded {len(existing_runs)} existing proposed run records from {raw_csv_path}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not parse existing {raw_csv_path}: {e}", flush=True)
                
    total_runs = len(datasets) * 4 * len(seeds)
    run_counter = 0
    t_start = time.time()
    
    discovered_equations: List[Dict[str, Any]] = []
    sample_trajectories: Dict[str, Any] = {}
    
    if verbose:
        print("=" * 115, flush=True)
        print(f" WD-GPR-SR Proposed Model LOCO Benchmark ({len(datasets)} datasets x 4 folds x {len(seeds)} seeds = {total_runs} runs)", flush=True)
        print("=" * 115, flush=True)
        
    for ds in datasets:
        ds_folds = splits_dict[ds]
        for test_cell, fold_data in ds_folds.items():
            for seed in seeds:
                run_counter += 1
                key = (ds, str(test_cell), int(seed))
                if key in existing_runs:
                    continue
                    
                t0 = time.time()
                try:
                    res = run_proposed_on_fold(fold_data, seed=seed)
                    m = res["metrics"]
                    m_gpr = res["metrics_wd_alone"]
                    elapsed = time.time() - t0
                    
                    rec = {
                        "dataset": ds,
                        "test_cell": test_cell,
                        "fold_idx": fold_data["fold_idx"],
                        "model": "WD-GPR-SR (Proposed)",
                        "family": "Proposed",
                        "seed": seed,
                        "MAE": m.get("MAE", np.nan),
                        "MSE": m.get("MSE", np.nan),
                        "RMSE": m.get("RMSE", np.nan),
                        "R2": m.get("R2", np.nan),
                        "MAPE": m.get("MAPE", np.nan),
                        "NLL": m.get("NLL", np.nan),
                        "PICP_95": m.get("PICP_95", np.nan),
                        "MPIW_95": m.get("MPIW_95", np.nan),
                        "Node_Count": m.get("Node_Count", np.nan),
                        "RMSE_GPR_Alone": m_gpr.get("RMSE", np.nan),
                        "R2_GPR_Alone": m_gpr.get("R2", np.nan),
                        "fit_time_s": round(res["fit_time"], 2),
                        "infer_time_s": round(res["infer_time"], 4),
                        "elapsed_s": round(elapsed, 2),
                        "equation": res["equation"],
                    }
                    raw_records.append(rec)
                    existing_runs.add(key)
                    
                    discovered_equations.append({
                        "dataset": ds,
                        "test_cell": test_cell,
                        "seed": seed,
                        "equation": res["equation"],
                        "RMSE": m["RMSE"],
                        "R2": m["R2"],
                    })
                    
                    # Store trajectory for seed 42 plotting
                    if seed == PRIMARY_SEED and test_cell not in sample_trajectories:
                        sample_trajectories[f"{ds}_{test_cell}"] = {
                            "dataset": ds,
                            "test_cell": test_cell,
                            "y_true": fold_data["y_test"],
                            "y_pred": res["y_pred"],
                            "y_var": res["y_var"],
                            "mu_wd": res["mu_wd"],
                            "equation": res["equation"],
                        }
                    
                    # Incremental write
                    if save_raw:
                        df_temp = pd.DataFrame(raw_records)
                        df_temp.to_csv(raw_csv_path, index=False)
                        
                    if verbose:
                        print(f"  [{run_counter:2d}/{total_runs}] {ds:<5} | Test: {test_cell:<8} | Seed: {seed:<4} | RMSE: {m['RMSE']:.4f}% | R²: {m['R2']:.4f} | NLL: {m['NLL']:.2f} | PICP: {m['PICP_95']:.1f}% ({elapsed:.1f}s)", flush=True)
                        print(f"       Discovered Residual Equation: {res['equation']}", flush=True)
                        
                except Exception as e:
                    print(f"  [ERROR] {ds} {test_cell} (seed {seed}): {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    
    df_raw = pd.DataFrame(raw_records)
    if save_raw:
        df_raw.to_csv(raw_csv_path, index=False)
        if verbose:
            print(f"\n[OK] Saved all {len(df_raw)} raw proposed run records to {raw_csv_path}", flush=True)
            
    # -----------------------------------------------------------------
    # Save Discovered Equations to Text File
    # -----------------------------------------------------------------
    eq_file = BASE_DIR / "results" / "discovered_equations.txt"
    with open(eq_file, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(" Discovered Closed-Form Residual Equations via Uncertainty-Weighted SR\n")
        f.write(" Author: Moonis Laskar (Roll No: 2401CS29) | Project: WD-GPR-SR\n")
        f.write("=" * 90 + "\n\n")
        for eq_item in discovered_equations:
            f.write(f"Dataset: {eq_item['dataset']} | Cell: {eq_item['test_cell']} | Seed: {eq_item['seed']}\n")
            f.write(f"  Performance: RMSE = {eq_item['RMSE']:.4f}%, R² = {eq_item['R2']:.4f}\n")
            f.write(f"  Equation   : {eq_item['equation']}\n\n")
    if verbose:
        print(f"[OK] Saved discovered symbolic equations to {eq_file}", flush=True)

    # -----------------------------------------------------------------
    # Aggregate Mean ± Std across 3 Seeds
    # -----------------------------------------------------------------
    metric_cols = ["RMSE", "MAE", "R2", "MAPE", "NLL", "PICP_95", "MPIW_95"]
    agg_rows = []
    
    if "dataset" not in df_raw.columns or len(df_raw) == 0:
        print("[WARN] No successful proposed runs to aggregate.", flush=True)
        return df_raw, pd.DataFrame(), sample_trajectories
        
    for ds in datasets:
        sub = df_raw[df_raw["dataset"] == ds]
        if len(sub) == 0:
            continue
        row = {
            "Dataset": ds,
            "Model": "WD-GPR-SR (Proposed)",
            "Family": "Proposed",
            "Total Runs": len(sub),
        }
        for m_col in metric_cols:
            vals = sub[m_col].dropna()
            if len(vals) > 0:
                mu = vals.mean()
                sd = vals.std()
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
        
    df_proposed_agg = pd.DataFrame(agg_rows)
    
    # -----------------------------------------------------------------
    # Update and Append to baselines_comparison.csv
    # -----------------------------------------------------------------
    baselines_csv = RESULTS_TAB_DIR / "baselines_comparison.csv"
    if append_to_baselines and baselines_csv.exists():
        try:
            df_base = pd.read_csv(baselines_csv)
            # Remove any previous proposed model entry to avoid duplicate rows
            df_base = df_base[df_base["Model"] != "WD-GPR-SR (Proposed)"]
            # Concatenate baseline table with proposed model rows
            df_master = pd.concat([df_base, df_proposed_agg], ignore_index=True)
            df_master.to_csv(baselines_csv, index=False)
            df_master.to_csv(RESULTS_TAB_DIR / "master_benchmark_comparison.csv", index=False)
            if verbose:
                print(f"[OK] Appended proposed model results to {baselines_csv} and saved master benchmark table", flush=True)
        except Exception as e:
            print(f"[WARN] Could not update {baselines_csv}: {e}", flush=True)
            df_master = df_proposed_agg
    else:
        df_master = df_proposed_agg
        
    total_time = time.time() - t_start
    if verbose:
        print(f"\nExecution finished in {total_time/60:.2f} minutes.")
        print_summary_table(df_proposed_agg)
        
    return df_raw, df_master, sample_trajectories


def print_summary_table(df_agg: pd.DataFrame):
    """Prints terminal summary table of proposed model performance."""
    print("\n" + "=" * 128)
    print(" WD-GPR-SR (Proposed) Evaluation Summary (Mean ± Std across 4 LOCO Folds x 3 Seeds)")
    print("=" * 128)
    print(f"{'Dataset':<8} {'Model':<24} {'Family':<14} {'RMSE (%)':<16} {'MAE (%)':<16} {'R²':<18} {'NLL':<14} {'PICP (95%)':<14}")
    print("-" * 128)
    for _, r in df_agg.iterrows():
        print(f"{r['Dataset']:<8} {r['Model']:<24} {r['Family']:<14} {r['RMSE']:<16} {r['MAE']:<16} {r['R2']:<18} {r['NLL']:<14} {r['PICP_95']:<14}")
    print("=" * 128)


# =====================================================================
# 3. Publication Plotting: Trajectories & Comparison Bar Charts
# =====================================================================

def plot_proposed_trajectories(
    trajectories: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plots degradation trajectories for representative NASA and CALCE cells,
    displaying true SOH, WD-GPR prior, full WD-GPR-SR prediction, and 95% confidence bands.
    """
    save_path = Path(save_path or (RESULTS_FIG_DIR / "soh_trajectory_proposed_predictions.png"))
    
    keys = list(trajectories.keys())
    if len(keys) == 0:
        return
        
    # Pick up to 4 representative cells
    plot_keys = keys[:4]
    fig, axes = plt.subplots(len(plot_keys), 1, figsize=(14, 4 * len(plot_keys)), dpi=300)
    if len(plot_keys) == 1:
        axes = [axes]
        
    for ax, k in zip(axes, plot_keys):
        traj = trajectories[k]
        y_true = np.asarray(traj["y_true"])
        y_pred = np.asarray(traj["y_pred"])
        mu_wd = np.asarray(traj["mu_wd"])
        sigma = np.sqrt(np.maximum(np.asarray(traj["y_var"]), 1e-8))
        cycles = np.arange(1, len(y_true) + 1)
        
        ax.plot(cycles, y_true, "k-", linewidth=2.0, label="Ground Truth SOH (%)")
        ax.plot(cycles, mu_wd, "b--", linewidth=1.5, alpha=0.8, label="Stage 1: WD-GPR Prior Mean")
        ax.plot(cycles, y_pred, "r-", linewidth=1.8, label="Stage 3: Full WD-GPR-SR Prediction")
        
        ax.fill_between(
            cycles,
            y_pred - 1.96 * sigma,
            y_pred + 1.96 * sigma,
            color="red",
            alpha=0.15,
            label="Calibrated 95% Confidence Band (±1.96σ)",
        )
        
        ax.set_title(
            f"Held-Out Cell: {traj['test_cell']} ({traj['dataset']}) | Discovered Residual: {traj['equation']}",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_xlabel("Cycle Index", fontsize=10)
        ax.set_ylabel("State of Health (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", frameon=True, fontsize=9)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved proposed trajectory predictions figure to {save_path}")


def plot_proposed_vs_baselines_barchart(
    df_master: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plots comparative bar charts highlighting the proposed model vs. all competitive baselines.
    """
    save_path = Path(save_path or (RESULTS_FIG_DIR / "proposed_vs_baselines_comparison.png"))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # 1. NASA RMSE
    ax1 = axes[0]
    sub_nasa = df_master[df_master["Dataset"] == "NASA"].sort_values("RMSE_mean")
    colors_nasa = ["#e41a1c" if "Proposed" in m else "#2b5c8f" for m in sub_nasa["Model"]]
    ax1.barh(sub_nasa["Model"], sub_nasa["RMSE_mean"], xerr=sub_nasa["RMSE_std"], color=colors_nasa, alpha=0.85, capsize=3)
    ax1.set_title("(a) NASA PCoE Benchmark: RMSE (SOH %)", fontweight="bold")
    ax1.set_xlabel("Root Mean Squared Error (%)")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax1.invert_yaxis()
    
    # 2. CALCE RMSE
    ax2 = axes[1]
    sub_calce = df_master[df_master["Dataset"] == "CALCE"].sort_values("RMSE_mean")
    colors_calce = ["#e41a1c" if "Proposed" in m else "#d95f02" for m in sub_calce["Model"]]
    ax2.barh(sub_calce["Model"], sub_calce["RMSE_mean"], xerr=sub_calce["RMSE_std"], color=colors_calce, alpha=0.85, capsize=3)
    ax2.set_title("(b) CALCE (UMD) Benchmark: RMSE (SOH %)", fontweight="bold")
    ax2.set_xlabel("Root Mean Squared Error (%)")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved proposed vs baselines comparison figure to {save_path}")


# =====================================================================
# Main Execution Runner
# =====================================================================

if __name__ == "__main__":
    df_raw, df_master, trajectories = evaluate_proposed_model(save_raw=True, append_to_baselines=True, verbose=True)
    plot_proposed_trajectories(trajectories)
    plot_proposed_vs_baselines_barchart(df_master)
