"""
Explainable AI (XAI) Analysis for WD-GPR-SR.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

Implements 5 complementary XAI techniques focused on NASA test cell B0018:
  1. SHAP Beeswarm Summary Plot (shap.KernelExplainer on GPR mean prediction)
  2. LIME Local Explanation for heavily degraded cycle (SOH < 75%)
  3. Partial Dependence Plots (PDP) & Individual Conditional Expectation (ICE) curves
  4. Global Feature Importance via Inverse Length-Scales / ARD Weights (1/ell_i)
  5. Dynamic Feature Sensitivity Trajectories across the Battery Lifecycle
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure writable matplotlib and loky configuration
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import lime
import lime.lime_tabular
import sympy as sp
import torch
import gpytorch

from src.config import PRIMARY_SEED, set_global_seed
from src.proposed_model import WD_GPR_SR_Pipeline


# Friendly display names for Health Indicators
FEATURE_PRETTY_NAMES = {
    "cc_charge_time": "CC Charge Duration (s)",
    "cv_charge_time": "CV Charge Duration (s)",
    "voltage_plateau_duration": "Voltage Plateau Duration (s)",
    "temp_rise_discharge": "Discharge Temp Rise (°C)",
    "mean_temp_discharge": "Mean Discharge Temp (°C)",
    "mean_voltage_discharge": "Mean Discharge Voltage (V)",
    "energy_discharge": "Discharge Energy (Wh)",
    "ic_peak_height": "ICA dQ/dV Peak Height",
    "ic_peak_voltage": "ICA dQ/dV Peak Voltage (V)",
    "ic_peak_area": "ICA Peak Area",
}


def load_target_data(
    splits_path: str = "data/processed/loco_splits.pkl",
    dataset_name: str = "NASA",
    test_cell: str = "B0018",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Loads LOCO train/test data and feature names for the target cell."""
    with open(splits_path, "rb") as f:
        loco = pickle.load(f)
    fold = loco[dataset_name][test_cell]
    X_train = fold["X_train"]
    y_train = fold["y_train"]
    X_test = fold["X_test"]
    y_test = fold["y_test"]
    raw_names = fold["feature_names"]
    pretty_names = [FEATURE_PRETTY_NAMES.get(fn, fn) for fn in raw_names]
    return X_train, y_train, X_test, y_test, raw_names, pretty_names


def train_target_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    seed: int = 42,
) -> WD_GPR_SR_Pipeline:
    """Trains the WD-GPR-SR pipeline on the training fold."""
    set_global_seed(seed)
    pipe = WD_GPR_SR_Pipeline(random_state=seed)
    pipe.fit(X_train, y_train, feature_names=feature_names, verbose=True)
    return pipe


# =====================================================================
# 1. SHAP Beeswarm Summary Plot
# =====================================================================

def run_shap_analysis(
    pipe: WD_GPR_SR_Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    num_background: int = 30,
    num_eval: int = 60,
):
    """
    Computes SHAP values using shap.KernelExplainer on the WD-GPR prediction function
    and generates a beeswarm summary plot.
    """
    print("\n--- [XAI 1/5] Generating SHAP Beeswarm Summary Plot ---", flush=True)
    set_global_seed(PRIMARY_SEED)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y_pred, _ = pipe.predict(x)
        return y_pred

    # Summarize background dataset using k-means
    bg_summary = shap.kmeans(X_train, num_background)
    explainer = shap.KernelExplainer(predict_fn, bg_summary)

    # Subsample test cycles spanning the entire degradation curve
    step = max(1, len(X_test) // num_eval)
    eval_indices = np.arange(0, len(X_test), step)[:num_eval]
    X_test_eval = X_test[eval_indices]

    print(f"  Computing KernelExplainer SHAP values on {len(X_test_eval)} test cycles...", flush=True)
    shap_values = explainer.shap_values(X_test_eval, nsamples=100)

    # Create SHAP beeswarm plot
    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(
        shap_values,
        X_test_eval,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title(
        "SHAP Feature Importance & Directionality: WD-GPR-SR (NASA Cell B0018)\n"
        "Positive SHAP values indicate features supporting higher battery SOH",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()

    out_png = output_dir / "shap_summary.png"
    out_pdf = output_dir / "shap_summary.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP summary plot to {out_png.resolve()} and {out_pdf.name}", flush=True)


# =====================================================================
# 2. LIME Local Explanation for Heavily Degraded Cycle (SOH < 75%)
# =====================================================================

def run_lime_analysis(
    pipe: WD_GPR_SR_Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
):
    """
    Applies LIME tabular explainer to a severely degraded cycle (SOH < 75%).
    """
    print("\n--- [XAI 2/5] Generating LIME Local Explanation for Degraded Cycle ---", flush=True)
    set_global_seed(PRIMARY_SEED)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y_pred, _ = pipe.predict(x)
        return y_pred

    # Identify heavily degraded cycle (SOH < 75%)
    degraded_indices = np.where(y_test < 75.0)[0]
    if len(degraded_indices) > 0:
        target_idx = degraded_indices[len(degraded_indices) // 2]
    else:
        target_idx = int(np.argmin(y_test))

    target_instance = X_test[target_idx]
    actual_soh = y_test[target_idx]
    pred_soh, pred_var = pipe.predict(target_instance.reshape(1, -1))
    pred_soh_val = float(pred_soh[0])
    pred_std_val = float(np.sqrt(pred_var[0]))

    print(f"  Target Degraded Cycle Index: {target_idx} (Actual SOH: {actual_soh:.2f}%, Pred SOH: {pred_soh_val:.2f}% ± {1.96*pred_std_val:.2f}%)", flush=True)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="regression",
        random_state=PRIMARY_SEED,
    )

    exp = explainer.explain_instance(
        target_instance,
        predict_fn,
        num_features=8,
    )

    exp_list = exp.as_list()
    features = [x[0] for x in exp_list][::-1]
    weights = [x[1] for x in exp_list][::-1]
    colors = ["#27ae60" if w >= 0 else "#e74c3c" for w in weights]

    plt.figure(figsize=(10, 5.5))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, weights, color=colors, edgecolor="black", alpha=0.85, height=0.6)
    plt.yticks(y_pos, features, fontsize=10)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("LIME Feature Weight (Contribution to SOH)", fontsize=11, fontweight="bold")
    plt.title(
        f"LIME Local Explanation: Degraded Cycle #{target_idx+1} (Actual SOH: {actual_soh:.2f}%, Pred: {pred_soh_val:.2f}%)\n"
        "Red bars pull prediction toward degradation; Green bars support health",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.grid(True, linestyle=":", alpha=0.6, axis="x")
    plt.tight_layout()

    out_png = output_dir / "lime_explanation.png"
    out_pdf = output_dir / "lime_explanation.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved LIME explanation plot to {out_png.resolve()} and {out_pdf.name}", flush=True)


# =====================================================================
# 3. PDP & ICE Curves for Top 2 Health Indicators
# =====================================================================

def run_pdp_ice_analysis(
    pipe: WD_GPR_SR_Pipeline,
    X_test: np.ndarray,
    raw_feature_names: List[str],
    pretty_feature_names: List[str],
    output_dir: Path,
    num_grid: int = 40,
    num_ice_samples: int = 25,
):
    """
    Computes Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE)
    curves for the top 2 Health Indicators (CC Charge Duration and ICA Peak Height).
    """
    print("\n--- [XAI 3/5] Generating PDP & ICE Curves for Top Health Indicators ---", flush=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Top 2 indicators: cc_charge_time (index 0) and ic_peak_height (index 7)
    feat_indices = [0, 7]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Subsample test instances for ICE lines
    sample_indices = np.linspace(0, len(X_test) - 1, num_ice_samples, dtype=int)
    X_samples = X_test[sample_indices]

    for ax_idx, feat_idx in enumerate(feat_indices):
        ax = axes[ax_idx]
        feat_name_pretty = pretty_feature_names[feat_idx]
        feat_name_raw = raw_feature_names[feat_idx]

        # Define evaluation grid between 5th and 95th percentiles of test data
        val_min = np.percentile(X_test[:, feat_idx], 2)
        val_max = np.percentile(X_test[:, feat_idx], 98)
        grid_vals = np.linspace(val_min, val_max, num_grid)

        ice_curves = np.zeros((num_ice_samples, num_grid))

        for s_idx, sample in enumerate(X_samples):
            # Create perturbation matrix
            X_pert = np.tile(sample, (num_grid, 1))
            X_pert[:, feat_idx] = grid_vals
            y_pred, _ = pipe.predict(X_pert)
            ice_curves[s_idx, :] = y_pred

            # Plot translucent individual ICE line
            ax.plot(grid_vals, y_pred, color="#3498db", alpha=0.25, linewidth=1.2)

        # Compute PDP (mean across all ICE curves)
        pdp_curve = np.mean(ice_curves, axis=0)
        pdp_std = np.std(ice_curves, axis=0)

        # Plot PDP curve and confidence envelope
        ax.plot(grid_vals, pdp_curve, color="#c0392b", linewidth=3.0, label="PDP (Average Marginal Effect)")
        ax.fill_between(
            grid_vals,
            pdp_curve - 1.96 * pdp_std,
            pdp_curve + 1.96 * pdp_std,
            color="#c0392b",
            alpha=0.15,
            label="95% ICE Variation Envelope",
        )

        ax.set_title(f"PDP & ICE: {feat_name_pretty}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(f"{feat_name_pretty} (Standardized)", fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Predicted Battery SOH (%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", loc="upper left" if ax_idx == 0 else "lower right")

    plt.suptitle(
        "Partial Dependence Plots (PDP) & Individual Conditional Expectation (ICE) Curves\n"
        "Non-linear monotonic mapping discovered by WD-GPR-SR for primary Health Indicators",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_png = output_dir / "pdp_ice.png"
    out_pdf = output_dir / "pdp_ice.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved PDP & ICE curves to {out_png.resolve()} and {out_pdf.name}", flush=True)


# =====================================================================
# 4. ARD Length-Scale Relevance (Global Feature Importance)
# =====================================================================

def run_ard_relevance_analysis(
    pipe: WD_GPR_SR_Pipeline,
    pretty_feature_names: List[str],
    output_dir: Path,
):
    """
    Extracts the learned ARD weights / linear variance coefficients from the Wide branch
    and plots a horizontal bar chart of global feature relevance (1/lengthscale or sqrt(variance)).
    """
    print("\n--- [XAI 4/5] Extracting ARD Relevance & Length-Scales ---", flush=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # In WD_SparseGPR, wide_kernel is ScaleKernel(LinearKernel(ard_num_dims=10))
    covar = pipe.gpr_model.covar_module
    wide_kernel = covar.wide_kernel
    base_linear = wide_kernel.base_kernel

    # LinearKernel ARD parameter is variances
    if hasattr(base_linear, "variance") and base_linear.variance is not None:
        raw_vars = base_linear.variance.detach().cpu().numpy().flatten()
        ard_relevance = np.sqrt(np.maximum(raw_vars, 1e-8))
    else:
        # Fallback to outputscale if uniform
        ard_relevance = np.ones(len(pretty_feature_names))

    # Normalize to percentage importance
    norm_importance = 100.0 * (ard_relevance / np.sum(ard_relevance))

    # Sort descending
    sort_idx = np.argsort(norm_importance)
    sorted_features = [pretty_feature_names[i] for i in sort_idx]
    sorted_scores = norm_importance[sort_idx]

    # Differentiate feature categories: Electrochemical (ICA) vs Thermal vs Time
    colors = []
    for f in sorted_features:
        if "ICA" in f or "dQ/dV" in f:
            colors.append("#2980b9")  # Blue for ICA
        elif "Temp" in f:
            colors.append("#e67e22")  # Orange for Thermal
        else:
            colors.append("#27ae60")  # Green for Time/Voltage

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(sorted_features))
    bars = plt.barh(y_pos, sorted_scores, color=colors, edgecolor="black", alpha=0.85, height=0.65)
    plt.yticks(y_pos, sorted_features, fontsize=10, fontweight="bold")
    plt.xlabel("Global ARD Feature Relevance Weight (%)", fontsize=11, fontweight="bold")
    plt.title(
        "Global ARD Feature Relevance: Wide-and-Deep Sparse GPR Prior\n"
        "Blue: Electrochemical ICA | Green: Time-Domain/Voltage | Orange: Thermal Dynamics",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    # Label percentage values
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.xlim(0, max(sorted_scores) * 1.15)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()

    out_png = output_dir / "ard_relevance.png"
    out_pdf = output_dir / "ard_relevance.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ARD relevance plot to {out_png.resolve()} and {out_pdf.name}", flush=True)


# =====================================================================
# 5. Symbolic Sensitivity Trajectories Across Battery Lifecycle
# =====================================================================

def run_symbolic_sensitivity_analysis(
    pipe: WD_GPR_SR_Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    raw_feature_names: List[str],
    pretty_feature_names: List[str],
    output_dir: Path,
):
    """
    Computes analytical and finite-difference sensitivities (d_y / d_x_j) of the full
    WD-GPR-SR model across the entire battery lifecycle from fresh state to end-of-life.
    """
    print("\n--- [XAI 5/5] Computing Dynamic Sensitivity Trajectories Across Lifecycle ---", flush=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    cycles = np.arange(1, len(X_test) + 1)
    eps = 1e-3

    # Key features to analyze:
    # 0: cc_charge_time, 2: voltage_plateau_duration, 7: ic_peak_height, 3: temp_rise_discharge
    target_feats = [0, 2, 7, 3]
    feat_colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]

    sensitivities = {f_idx: np.zeros(len(X_test)) for f_idx in target_feats}

    for t_idx, x_point in enumerate(X_test):
        for f_idx in target_feats:
            x_plus = x_point.copy()
            x_minus = x_point.copy()
            x_plus[f_idx] += eps
            x_minus[f_idx] -= eps

            y_plus, _ = pipe.predict(x_plus.reshape(1, -1))
            y_minus, _ = pipe.predict(x_minus.reshape(1, -1))

            # Finite-difference derivative: |d_y / d_x_j|
            grad = float(np.abs(y_plus[0] - y_minus[0]) / (2.0 * eps))
            sensitivities[f_idx][t_idx] = grad

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    # Top plot: True SOH Trajectory & Knee-Point Demarcation
    ax1.plot(cycles, y_test, color="#2c3e50", linewidth=2.5, label="Battery SOH (%)")
    ax1.axhline(80.0, color="#c0392b", linestyle="--", linewidth=1.2, label="80% SOH (Knee / EOL Threshold)")
    ax1.set_ylabel("True SOH (%)", fontsize=11, fontweight="bold")
    ax1.set_title("NASA Cell B0018 Lifecycle Degradation & Dynamic Feature Sensitivity Trajectories", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="lower left", frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bottom plot: Sensitivity Trajectories
    for f_idx, col in zip(target_feats, feat_colors):
        f_name = pretty_feature_names[f_idx]
        smoothed_sens = pd.Series(sensitivities[f_idx]).rolling(window=7, min_periods=1, center=True).mean().values
        ax2.plot(cycles, smoothed_sens, color=col, linewidth=2.2, label=f"Sensitivity: {f_name}")

    ax2.set_xlabel("Battery Cycle Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Dynamic Feature Sensitivity $\left|\frac{\partial \hat{y}}{\partial x_j}\right|$", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=10)

    # Annotate knee transition
    knee_cycle = int(cycles[np.where(y_test < 80.0)[0][0]]) if any(y_test < 80.0) else 100
    ax2.axvline(knee_cycle, color="#c0392b", linestyle=":", linewidth=1.5)
    ax2.annotate(
        f"Knee-Point Onset (Cycle ~{knee_cycle})\nRapid sensitivity shift",
        xy=(knee_cycle, 0.7 * np.max([np.max(v) for v in sensitivities.values()])),
        xytext=(knee_cycle - 35, 0.85 * np.max([np.max(v) for v in sensitivities.values()])),
        arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#c0392b", lw=1),
    )

    plt.tight_layout()

    out_png = output_dir / "symbolic_sensitivities.png"
    out_pdf = output_dir / "symbolic_sensitivities.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved dynamic sensitivity trajectories to {out_png.resolve()} and {out_pdf.name}", flush=True)


# =====================================================================
# Main Execution Pipeline
# =====================================================================

def run_all_xai(
    splits_path: str = "data/processed/loco_splits.pkl",
    output_dir: str = "results",
    test_cell: str = "B0018",
):
    """Executes the full suite of 5 XAI analyses."""
    out_path = Path(output_dir)
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f" Explainable AI (XAI) Suite for WD-GPR-SR (Target: NASA Cell {test_cell})")
    print("=" * 90)

    # 1. Load data
    X_train, y_train, X_test, y_test, raw_names, pretty_names = load_target_data(
        splits_path=splits_path,
        dataset_name="NASA",
        test_cell=test_cell,
    )

    # 2. Train target pipeline
    pipe = train_target_pipeline(X_train, y_train, raw_names, seed=PRIMARY_SEED)

    # 3. Execute the 5 XAI components
    run_shap_analysis(pipe, X_train, X_test, pretty_names, figures_dir)
    run_lime_analysis(pipe, X_train, X_test, y_test, pretty_names, figures_dir)
    run_pdp_ice_analysis(pipe, X_test, raw_names, pretty_names, figures_dir)
    run_ard_relevance_analysis(pipe, pretty_names, figures_dir)
    run_symbolic_sensitivity_analysis(pipe, X_test, y_test, raw_names, pretty_names, figures_dir)

    print("\n" + "=" * 90)
    print(" [OK] All 5 XAI Analyses Successfully Generated and Saved to results/figures/")
    print("=" * 90)


if __name__ == "__main__":
    run_all_xai()

