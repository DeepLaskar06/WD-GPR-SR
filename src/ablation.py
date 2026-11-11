"""
Ablation Study: Empirical Decomposition of WD-GPR-SR Architecture.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

Evaluates 4 model variants across Leave-One-Cell-Out (LOCO) splits and 3 global seeds [42, 100, 2026]:
  - Variant A (Deep-Only GPR): Deep Kernel Learning (MLP + Matérn) without Wide/Linear branch and without SR.
  - Variant B (WD-GPR without SR): Proposed Wide-and-Deep Sparse GPR, skipping residual Symbolic Regression.
  - Variant C (WD-GPR + Unweighted SR): Proposed WD-GPR prior + residual SR trained with standard unweighted MSE.
  - Variant D (Full WD-GPR-SR): Complete proposed model with heteroscedastic inverse-variance weighting (1/sigma^2).
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure writable matplotlib and loky configuration
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from sklearn.cluster import KMeans

from src.config import SEEDS, PRIMARY_SEED, set_global_seed
from src.metrics import evaluate_predictions
from src.setup_env import UncertaintyWeightedSR
from src.proposed_model import (
    DeepFeatureExtractor,
    WDCovariance,
    WD_SparseGPR,
    train_wd_sparse_gpr,
    WD_GPR_SR_Pipeline,
)


# =====================================================================
# Variant A: Deep-Only Sparse GPR (No Wide Branch, No SR)
# =====================================================================

class DeepOnlyCovariance(gpytorch.kernels.Kernel):
    """
    Variant A Covariance:
      k_DeepOnly(x, x') = k_Deep(phi(x), phi(x')) + k_stabilizer(x, x')
    Excludes the Wide linear branch to isolate deep representation contribution.
    """
    def __init__(self, feature_extractor: nn.Module, deep_dim: int = 8):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.deep_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=deep_dim)
        )
        self.stabilizer_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.stabilizer_kernel.outputscale = 1e-3

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        phi_x1 = self.feature_extractor(x1)
        phi_x2 = self.feature_extractor(x2) if (x2 is not x1 and not diag) else phi_x1
        k_deep = self.deep_kernel(phi_x1, phi_x2, diag=diag, **params)
        k_stab = self.stabilizer_kernel(x1, x2, diag=diag, **params)
        return k_deep + k_stab


class DeepOnly_SparseGPR(gpytorch.models.ApproximateGP):
    """
    Sparse Variational GPR without Wide Linear branch.
    """
    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_dim: int = 10,
        deep_dim: int = 8,
        y_mean: float = 85.0,
    ):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant.data.fill_(y_mean)
        self.feature_extractor = DeepFeatureExtractor(input_dim=input_dim, out_dim=deep_dim)
        self.covar_module = DeepOnlyCovariance(
            self.feature_extractor,
            deep_dim=deep_dim,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_deep_only_sparse_gpr(
    model: DeepOnly_SparseGPR,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.03,
) -> Tuple[DeepOnly_SparseGPR, gpytorch.likelihoods.GaussianLikelihood]:
    """Trains Deep-Only Sparse GPR via Variational ELBO."""
    t_x = torch.tensor(X_train, dtype=torch.float32)
    t_y = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(t_x, t_y), batch_size=batch_size, shuffle=True)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=t_y.size(0))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    with gpytorch.settings.cholesky_jitter(1e-3):
        model.train()
        likelihood.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                output = model(bx)
                loss = -mll(output, by)
                loss.backward()
                optimizer.step()
            scheduler.step()
    return model, likelihood


# =====================================================================
# Ablation Execution Harness
# =====================================================================

def run_ablation_study(
    splits_path: str = "data/processed/loco_splits.pkl",
    output_dir: str = "results",
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Executes the 4 ablation variants across LOCO splits and seeds.
    """
    if seeds is None:
        seeds = SEEDS
    if datasets is None:
        datasets = ["NASA", "CALCE"]

    out_path = Path(output_dir)
    tables_dir = out_path / "tables"
    figures_dir = out_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = tables_dir / "ablation_raw_runs.csv"
    summary_csv = tables_dir / "ablation_study.csv"

    # Load existing raw records if available for auto-resume
    existing_runs = set()
    records: List[Dict[str, Any]] = []
    if raw_csv.exists():
        try:
            df_existing = pd.read_csv(raw_csv)
            records = df_existing.to_dict("records")
            for r in records:
                existing_runs.add((r["Dataset"], r["Test_Cell"], int(r["Seed"]), r["Variant"]))
            print(f"[OK] Found {len(records)} existing ablation runs in {raw_csv.name}. Resuming...", flush=True)
        except Exception as e:
            print(f"[!] Warning reading {raw_csv}: {e}", flush=True)

    # Check if proposed raw runs can supply Variant D
    proposed_raw_csv = tables_dir / "proposed_raw_runs.csv"
    proposed_lookup = {}
    if proposed_raw_csv.exists():
        try:
            df_prop = pd.read_csv(proposed_raw_csv)
            for _, row in df_prop.iterrows():
                ds_val = str(row.get("dataset", row.get("Dataset", "")))
                tc_val = str(row.get("test_cell", row.get("Test_Cell", "")))
                sd_val = int(row.get("seed", row.get("Seed", 42)))
                proposed_lookup[(ds_val, tc_val, sd_val)] = row
            print(f"[OK] Pre-loaded {len(proposed_lookup)} proposed model runs for Variant D reference.", flush=True)
        except Exception as e:
            print(f"[!] Could not pre-load proposed runs: {e}", flush=True)

    with open(splits_path, "rb") as f:
        loco_data = pickle.load(f)

    variant_names = [
        ("Variant A (Deep-Only GPR)", "Deep-Only Sparse GPR (no Wide branch, no SR)"),
        ("Variant B (WD-GPR without SR)", "Wide-and-Deep Sparse GPR (no SR correction)"),
        ("Variant C (WD-GPR + Unweighted SR)", "Wide-and-Deep Sparse GPR + Unweighted SR"),
        ("Variant D (Full WD-GPR-SR)", "Full WD-GPR-SR (Uncertainty-Weighted 1/σ²)"),
    ]

    total_tasks = len(datasets) * 4 * len(seeds)
    print("\n" + "=" * 115)
    print(" Ablation Study: 4 Model Variants Across LOCO Folds (Author: Moonis Laskar, 2401CS29)")
    print("=" * 115)

    task_idx = 0
    for ds_name in datasets:
        if ds_name not in loco_data:
            continue
        folds = loco_data[ds_name]
        cell_keys = sorted(folds.keys())

        for cell_id in cell_keys:
            fold = folds[cell_id]
            X_train = fold["X_train"]
            y_train = fold["y_train"]
            X_test = fold["X_test"]
            y_test = fold["y_test"]
            feature_names = fold["feature_names"]
            input_dim = X_train.shape[1]
            deep_dim = 8
            y_mean = float(y_train.mean())

            for seed in seeds:
                task_idx += 1
                prefix = f"[{task_idx:02d}/{total_tasks:02d}] {ds_name:<5} | Test: {cell_id:<8} | Seed: {seed:<4}"

                # -------------------------------------------------------------
                # 1. Variant A (Deep-Only GPR)
                # -------------------------------------------------------------
                v_a_key = (ds_name, cell_id, seed, "Variant A (Deep-Only GPR)")
                if v_a_key not in existing_runs:
                    t0 = time.time()
                    set_global_seed(seed)
                    kmeans = KMeans(n_clusters=min(64, len(X_train)), random_state=seed, n_init=1).fit(X_train)
                    inducing_pts = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

                    model_a = DeepOnly_SparseGPR(
                        inducing_points=inducing_pts,
                        input_dim=input_dim,
                        deep_dim=deep_dim,
                        y_mean=y_mean,
                    )
                    lik_a = gpytorch.likelihoods.GaussianLikelihood()
                    model_a, lik_a = train_deep_only_sparse_gpr(
                        model_a, lik_a, X_train, y_train, epochs=35, batch_size=64, lr=0.03
                    )

                    model_a.eval()
                    lik_a.eval()
                    t_xtest = torch.tensor(X_test, dtype=torch.float32)
                    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
                        dist_a = lik_a(model_a(t_xtest))
                        pred_a = dist_a.mean.cpu().numpy()
                        var_a = dist_a.variance.cpu().numpy()

                    m_a = evaluate_predictions(y_test, pred_a, y_var=var_a)
                    rec_a = {
                        "Dataset": ds_name,
                        "Test_Cell": cell_id,
                        "Seed": seed,
                        "Variant": "Variant A (Deep-Only GPR)",
                        "Description": "Deep Kernel Learning (MLP + Matérn, no Wide, no SR)",
                        "Elapsed_Sec": round(time.time() - t0, 2),
                        **m_a,
                    }
                    records.append(rec_a)
                    existing_runs.add(v_a_key)
                    pd.DataFrame(records).to_csv(raw_csv, index=False)
                    print(f"{prefix} | Var A (Deep-Only)       | RMSE: {m_a['RMSE']:.4f}% | R²: {m_a['R2']:.4f} | PICP: {m_a['PICP_95']:.1f}% ({rec_a['Elapsed_Sec']}s)", flush=True)

                # -------------------------------------------------------------
                # 2. Train Stage 1 WD-SparseGPR (Shared for Variant B, C, D)
                # -------------------------------------------------------------
                needs_stage1 = (
                    (ds_name, cell_id, seed, "Variant B (WD-GPR without SR)") not in existing_runs or
                    (ds_name, cell_id, seed, "Variant C (WD-GPR + Unweighted SR)") not in existing_runs or
                    ((ds_name, cell_id, seed, "Variant D (Full WD-GPR-SR)") not in existing_runs and (ds_name, cell_id, seed) not in proposed_lookup)
                )

                if needs_stage1:
                    set_global_seed(seed)
                    kmeans_wd = KMeans(n_clusters=min(64, len(X_train)), random_state=seed, n_init=1).fit(X_train)
                    inducing_wd = torch.tensor(kmeans_wd.cluster_centers_, dtype=torch.float32)

                    model_wd = WD_SparseGPR(
                        inducing_points=inducing_wd,
                        input_dim=input_dim,
                        deep_dim=deep_dim,
                        y_mean=y_mean,
                    )
                    lik_wd = gpytorch.likelihoods.GaussianLikelihood()
                    model_wd, lik_wd, _ = train_wd_sparse_gpr(
                        model_wd, lik_wd, X_train, y_train, epochs=35, batch_size=64, lr=0.03
                    )

                    model_wd.eval()
                    lik_wd.eval()
                    t_xtest = torch.tensor(X_test, dtype=torch.float32)
                    t_xtrain = torch.tensor(X_train, dtype=torch.float32)

                    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
                        dist_test = lik_wd(model_wd(t_xtest))
                        mu_test = dist_test.mean.cpu().numpy()
                        var_test = dist_test.variance.cpu().numpy()

                        dist_train = lik_wd(model_wd(t_xtrain))
                        mu_train = dist_train.mean.cpu().numpy()
                        var_train = dist_train.variance.cpu().numpy()

                    residuals_train = y_train - mu_train
                    res_std = float(residuals_train.std())
                    max_dev = 3.0 * max(res_std, 0.5)

                    # ---------------------------------------------------------
                    # Variant B (WD-GPR without SR)
                    # ---------------------------------------------------------
                    v_b_key = (ds_name, cell_id, seed, "Variant B (WD-GPR without SR)")
                    if v_b_key not in existing_runs:
                        m_b = evaluate_predictions(y_test, mu_test, y_var=var_test)
                        rec_b = {
                            "Dataset": ds_name,
                            "Test_Cell": cell_id,
                            "Seed": seed,
                            "Variant": "Variant B (WD-GPR without SR)",
                            "Description": "Wide-and-Deep Sparse GPR (no SR correction)",
                            "Elapsed_Sec": 0.0,
                            **m_b,
                        }
                        records.append(rec_b)
                        existing_runs.add(v_b_key)
                        pd.DataFrame(records).to_csv(raw_csv, index=False)
                        print(f"{prefix} | Var B (WD-GPR no SR)     | RMSE: {m_b['RMSE']:.4f}% | R²: {m_b['R2']:.4f} | PICP: {m_b['PICP_95']:.1f}%", flush=True)

                    # ---------------------------------------------------------
                    # Variant C (WD-GPR + Unweighted SR)
                    # ---------------------------------------------------------
                    v_c_key = (ds_name, cell_id, seed, "Variant C (WD-GPR + Unweighted SR)")
                    if v_c_key not in existing_runs:
                        t0 = time.time()
                        set_global_seed(seed)
                        sr_unweighted = UncertaintyWeightedSR(
                            population_size=200,
                            generations=10,
                            random_state=seed,
                            parsimony_coefficient=0.005,
                        )
                        # sigma_sq=None -> standard unweighted MSE
                        sr_unweighted.fit(X_train, residuals_train, sigma_sq=None, feature_names=feature_names)
                        res_pred_c = np.clip(sr_unweighted.predict(X_test), -max_dev, max_dev)
                        y_pred_c = mu_test + res_pred_c

                        m_c = evaluate_predictions(y_test, y_pred_c, y_var=var_test)
                        rec_c = {
                            "Dataset": ds_name,
                            "Test_Cell": cell_id,
                            "Seed": seed,
                            "Variant": "Variant C (WD-GPR + Unweighted SR)",
                            "Description": "WD-GPR + Unweighted Symbolic Residual Regression",
                            "Elapsed_Sec": round(time.time() - t0, 2),
                            **m_c,
                        }
                        records.append(rec_c)
                        existing_runs.add(v_c_key)
                        pd.DataFrame(records).to_csv(raw_csv, index=False)
                        print(f"{prefix} | Var C (WD-GPR + Unweighted SR)| RMSE: {m_c['RMSE']:.4f}% | R²: {m_c['R2']:.4f} | PICP: {m_c['PICP_95']:.1f}% ({rec_c['Elapsed_Sec']}s)", flush=True)

                # -------------------------------------------------------------
                # 3. Variant D (Full WD-GPR-SR)
                # -------------------------------------------------------------
                v_d_key = (ds_name, cell_id, seed, "Variant D (Full WD-GPR-SR)")
                if v_d_key not in existing_runs:
                    prop_key = (ds_name, cell_id, seed)
                    if prop_key in proposed_lookup:
                        # Pull pre-computed record from proposed_raw_runs.csv
                        p_row = proposed_lookup[prop_key]
                        rec_d = {
                            "Dataset": ds_name,
                            "Test_Cell": cell_id,
                            "Seed": seed,
                            "Variant": "Variant D (Full WD-GPR-SR)",
                            "Description": "Complete Proposed Model (WD-GPR + 1/σ² Weighted SR)",
                            "Elapsed_Sec": float(p_row.get("Elapsed_Sec", p_row.get("elapsed_s", 0.0))),
                            "MAE": p_row["MAE"],
                            "MSE": p_row["MSE"],
                            "RMSE": p_row["RMSE"],
                            "R2": p_row["R2"],
                            "MAPE": p_row["MAPE"],
                            "NLL": p_row["NLL"],
                            "PICP_95": p_row["PICP_95"],
                            "MPIW_95": p_row["MPIW_95"],
                        }
                        records.append(rec_d)
                        existing_runs.add(v_d_key)
                        pd.DataFrame(records).to_csv(raw_csv, index=False)
                        print(f"{prefix} | Var D (Full WD-GPR-SR)   | RMSE: {rec_d['RMSE']:.4f}% | R²: {rec_d['R2']:.4f} | PICP: {rec_d['PICP_95']:.1f}% (Pre-computed)", flush=True)
                    else:
                        t0 = time.time()
                        pipe = WD_GPR_SR_Pipeline(random_state=seed)
                        pipe.fit(X_train, y_train, feature_names=feature_names)
                        pred_d, var_d = pipe.predict(X_test)
                        m_d = evaluate_predictions(y_test, pred_d, y_var=var_d)
                        rec_d = {
                            "Dataset": ds_name,
                            "Test_Cell": cell_id,
                            "Seed": seed,
                            "Variant": "Variant D (Full WD-GPR-SR)",
                            "Description": "Complete Proposed Model (WD-GPR + 1/σ² Weighted SR)",
                            "Elapsed_Sec": round(time.time() - t0, 2),
                            **m_d,
                        }
                        records.append(rec_d)
                        existing_runs.add(v_d_key)
                        pd.DataFrame(records).to_csv(raw_csv, index=False)
                        print(f"{prefix} | Var D (Full WD-GPR-SR)   | RMSE: {m_d['RMSE']:.4f}% | R²: {m_d['R2']:.4f} | PICP: {m_d['PICP_95']:.1f}% ({rec_d['Elapsed_Sec']}s)", flush=True)

    # =========================================================================
    # Aggregation & Summary Table Construction
    # =========================================================================
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(raw_csv, index=False)

    summary_rows = []
    variant_order = [
        "Variant A (Deep-Only GPR)",
        "Variant B (WD-GPR without SR)",
        "Variant C (WD-GPR + Unweighted SR)",
        "Variant D (Full WD-GPR-SR)",
    ]

    metric_cols = ["RMSE", "MAE", "MSE", "R2", "MAPE", "NLL", "PICP_95", "MPIW_95"]

    for ds_name in datasets:
        ds_slice = df_raw[df_raw["Dataset"] == ds_name]
        for v_name in variant_order:
            sub = ds_slice[ds_slice["Variant"] == v_name]
            if len(sub) == 0:
                continue

            row_dict = {
                "Dataset": ds_name,
                "Variant": v_name,
                "Total Runs": len(sub),
            }

            for col in metric_cols:
                m_mean = sub[col].mean()
                m_std = sub[col].std()
                if col in ["RMSE", "MAE", "MAPE"]:
                    row_dict[col] = f"{m_mean:.3f} ± {m_std:.3f}"
                elif col in ["MSE"]:
                    row_dict[col] = f"{m_mean:.4f} ± {m_std:.4f}"
                elif col in ["R2"]:
                    row_dict[col] = f"{m_mean:.4f} ± {m_std:.4f}"
                elif col in ["NLL"]:
                    row_dict[col] = f"{m_mean:.2f} ± {m_std:.2f}"
                elif col in ["PICP_95"]:
                    row_dict[col] = f"{m_mean:.1f} ± {m_std:.1f}"
                elif col in ["MPIW_95"]:
                    row_dict[col] = f"{m_mean:.1f} ± {m_std:.1f}"

                row_dict[f"{col}_mean"] = m_mean
                row_dict[f"{col}_std"] = m_std

            summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n[OK] Saved aggregated ablation study table to {summary_csv.resolve()}", flush=True)

    # Print summary to stdout
    print("\n" + "=" * 128)
    print(" Ablation Study Summary: Architecture Stepwise Impact (Mean ± Std Across LOCO Folds & 3 Seeds)")
    print("=" * 128)
    disp_cols = ["Dataset", "Variant", "RMSE", "MAE", "R2", "NLL", "PICP_95", "MPIW_95"]
    print(df_summary[disp_cols].to_string(index=False))
    print("=" * 128)

    # =========================================================================
    # High-Resolution Publication Figure Generation (300 DPI)
    # =========================================================================
    generate_ablation_figure(df_summary, figures_dir)

    return df_summary


def generate_ablation_figure(df_summary: pd.DataFrame, figures_dir: Path):
    """
    Generates a publication-grade ablation comparison figure showing the stepwise
    reduction in RMSE and MAE alongside calibrated PICP coverage.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#9b59b6", "#3498db", "#e67e22", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    variants_clean = [
        "A: Deep-Only GPR",
        "B: WD-GPR (No SR)",
        "C: WD-GPR + Unwt SR",
        "D: Full WD-GPR-SR",
    ]

    for ax_idx, ds_name in enumerate(["NASA", "CALCE"]):
        ax = axes[ax_idx]
        sub = df_summary[df_summary["Dataset"] == ds_name]
        if len(sub) == 0:
            continue

        rmse_means = sub["RMSE_mean"].values
        rmse_stds = sub["RMSE_std"].values
        mae_means = sub["MAE_mean"].values
        mae_stds = sub["MAE_std"].values

        x = np.arange(len(sub))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            rmse_means,
            width,
            yerr=rmse_stds,
            capsize=4,
            label="RMSE (%)",
            color="#2980b9",
            alpha=0.9,
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + width / 2,
            mae_means,
            width,
            yerr=mae_stds,
            capsize=4,
            label="MAE (%)",
            color="#27ae60",
            alpha=0.9,
            edgecolor="black",
        )

        # Label values on top of bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05 * (max(rmse_means) + 0.1), f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05 * (max(rmse_means) + 0.1), f"{h:.2f}%", ha="center", va="bottom", fontsize=9)

        ax.set_title(f"{ds_name} Dataset: Architectural Stepwise Ablation", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(variants_clean[:len(sub)], fontsize=10, fontweight="bold")
        ax.set_ylabel("Error Metric (%)", fontsize=11, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Ablation Study: Progressive Architectural Impact on Battery SOH Prediction\n"
        "Variant A (Deep-Only) → Variant B (Wide-and-Deep) → Variant C (+ Unweighted SR) → Variant D (Full WD-GPR-SR)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_png = figures_dir / "ablation_study_comparison.png"
    out_pdf = figures_dir / "ablation_study_comparison.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ablation comparison figure to {out_png.resolve()} and {out_pdf.name}", flush=True)


if __name__ == "__main__":
    run_ablation_study()
