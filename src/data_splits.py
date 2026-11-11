"""
Strict Leave-One-Cell-Out (LOCO) Cross-Validation and Data Leakage Prevention Pipeline.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

This module implements:
  1. Strict Leave-One-Cell-Out (LOCO) cross-validation per benchmark dataset (NASA / CALCE).
  2. Chronological within-cell validation split (first 85% train, final 15% val) with zero data leakage.
  3. Strict isolation of sklearn.preprocessing.StandardScaler fit exclusively on X_train.
  4. 3D sliding-window tensor generation (N, window_size, n_features) for sequence baselines (BiLSTM),
     aligned 1-to-1 with tabular 2D targets.
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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from src.config import (
    DATA_PROCESSED_DIR,
    RESULTS_FIG_DIR,
    RESULTS_TAB_DIR,
    NASA_CELLS,
    CALCE_CELLS,
    PRIMARY_SEED,
)

FEATURE_COLS = [
    "cc_charge_time",
    "cv_charge_time",
    "voltage_plateau_duration",
    "temp_rise_discharge",
    "mean_temp_discharge",
    "mean_voltage_discharge",
    "energy_discharge",
    "ic_peak_height",
    "ic_peak_voltage",
    "ic_peak_area",
]

TARGET_COL = "soh"


# =====================================================================
# 1. 3D Sequence Tensor Constructor (BiLSTM Support)
# =====================================================================

def make_cell_sequences(
    X_cell: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """
    Constructs causal 3D sliding-window sequences (N_samples, window_size, n_features)
    for a single cell. Pre-pads with the initial cycle's feature vector for cycles
    where t < window_size so that sample count matches 2D tabular features exactly.
    
    Guarantees:
      1. Zero boundary crossing: sequences never mix across cells.
      2. Perfect alignment: seq[i, -1, :] == X_cell[i, :].
      3. Identical target alignment between tabular and sequence models.
    """
    N, D = X_cell.shape
    pad_len = window_size - 1
    
    if pad_len > 0:
        first_row = X_cell[0:1]  # shape (1, D)
        padding = np.repeat(first_row, pad_len, axis=0)
        X_padded = np.vstack([padding, X_cell])
    else:
        X_padded = X_cell
        
    seqs = np.zeros((N, window_size, D), dtype=np.float32)
    for i in range(N):
        seqs[i] = X_padded[i : i + window_size]
        
    return seqs


# =====================================================================
# 2. Leave-One-Cell-Out (LOCO) Split Engine
# =====================================================================

def get_loco_splits(
    df: pd.DataFrame,
    dataset_name: str = "NASA",
    val_ratio: float = 0.15,
    window_size: int = 10,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs strict Leave-One-Cell-Out (LOCO) folds for a given dataset.
    
    Parameters:
        df          : Full feature DataFrame (e.g. from features_all.csv).
        dataset_name: 'NASA' or 'CALCE'.
        val_ratio   : Proportion of cycles at the end of each training cell reserved for validation (0.15).
        window_size : Sequence history length for 3D tensor creation.
        feature_cols: List of feature names (defaults to 10 HIs).
        target_col  : Prediction target (defaults to 'soh').
        
    Returns:
        Dictionary mapping test_cell -> fold dictionary containing:
          - X_train, y_train, X_train_seq, X_train_raw
          - X_val, y_val, X_val_seq, X_val_raw
          - X_test, y_test, X_test_seq, X_test_raw
          - scaler (StandardScaler fitted strictly on X_train_raw)
          - meta_train, meta_val, meta_test (cycle metadata)
    """
    features = feature_cols or FEATURE_COLS
    df_sub = df[df["dataset"] == dataset_name].copy()
    
    cells = sorted(df_sub["cell_id"].unique())
    if len(cells) < 2:
        raise ValueError(f"Dataset '{dataset_name}' must contain at least 2 cells for LOCO cross-validation.")
        
    folds: Dict[str, Dict[str, Any]] = {}
    
    for fold_idx, test_cell in enumerate(cells):
        train_cells = [c for c in cells if c != test_cell]
        
        cell_train_raw_list = []
        cell_val_raw_list = []
        cell_train_y_list = []
        cell_val_y_list = []
        cell_train_meta_list = []
        cell_val_meta_list = []
        
        # 1. Split training cells chronologically
        for c in train_cells:
            sub = df_sub[df_sub["cell_id"] == c].sort_values("cycle_idx").reset_index(drop=True)
            n_tot = len(sub)
            n_train = int(np.floor(n_tot * (1.0 - val_ratio)))
            
            sub_train = sub.iloc[:n_train]
            sub_val = sub.iloc[n_train:]
            
            cell_train_raw_list.append(sub_train[features].values.astype(np.float64))
            cell_val_raw_list.append(sub_val[features].values.astype(np.float64))
            cell_train_y_list.append(sub_train[target_col].values.astype(np.float64))
            cell_val_y_list.append(sub_val[target_col].values.astype(np.float64))
            
            cell_train_meta_list.append(sub_train[["cell_id", "cycle_idx", "soh", "capacity"]])
            cell_val_meta_list.append(sub_val[["cell_id", "cycle_idx", "soh", "capacity"]])
            
        # 2. Held-out test cell (all cycles)
        sub_test = df_sub[df_sub["cell_id"] == test_cell].sort_values("cycle_idx").reset_index(drop=True)
        X_test_raw = sub_test[features].values.astype(np.float64)
        y_test = sub_test[target_col].values.astype(np.float64)
        meta_test = sub_test[["cell_id", "cycle_idx", "soh", "capacity"]].copy()
        
        # Combine 2D training and validation partitions
        X_train_raw = np.vstack(cell_train_raw_list)
        y_train = np.concatenate(cell_train_y_list)
        meta_train = pd.concat(cell_train_meta_list, ignore_index=True)
        
        X_val_raw = np.vstack(cell_val_raw_list)
        y_val = np.concatenate(cell_val_y_list)
        meta_val = pd.concat(cell_val_meta_list, ignore_index=True)
        
        # 3. Fit StandardScaler STRICTLY on X_train_raw (Zero Data Leakage)
        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
        
        # 4. Construct 3D sliding window tensors per cell using normalized features
        cell_train_seq_list = []
        cell_val_seq_list = []
        
        for c in train_cells:
            sub = df_sub[df_sub["cell_id"] == c].sort_values("cycle_idx").reset_index(drop=True)
            n_tot = len(sub)
            n_train = int(np.floor(n_tot * (1.0 - val_ratio)))
            
            X_c_norm = scaler.transform(sub[features].values.astype(np.float64))
            seq_c = make_cell_sequences(X_c_norm, window_size=window_size)
            
            cell_train_seq_list.append(seq_c[:n_train])
            cell_val_seq_list.append(seq_c[n_train:])
            
        X_train_seq = np.vstack(cell_train_seq_list)
        X_val_seq = np.vstack(cell_val_seq_list)
        
        # Test sequence constructed from normalized test cell
        X_test_seq = make_cell_sequences(X_test, window_size=window_size)
        
        # Rigorous consistency assertions
        assert len(X_train) == len(X_train_seq) == len(y_train), "Train dimension mismatch!"
        assert len(X_val) == len(X_val_seq) == len(y_val), "Val dimension mismatch!"
        assert len(X_test) == len(X_test_seq) == len(y_test), "Test dimension mismatch!"
        assert np.allclose(X_train_seq[:, -1, :], X_train, atol=1e-5), "Train sequence alignment mismatch!"
        assert np.allclose(X_val_seq[:, -1, :], X_val, atol=1e-5), "Val sequence alignment mismatch!"
        assert np.allclose(X_test_seq[:, -1, :], X_test, atol=1e-5), "Test sequence alignment mismatch!"
        assert not np.isnan(X_train).any() and not np.isnan(X_val).any() and not np.isnan(X_test).any(), "Found NaNs in splits!"
        
        folds[test_cell] = {
            "fold_idx": fold_idx,
            "dataset": dataset_name,
            "test_cell": test_cell,
            "train_cells": train_cells,
            "feature_names": features,
            "target_name": target_col,
            "scaler": scaler,
            "X_train": X_train,
            "y_train": y_train,
            "X_train_seq": X_train_seq,
            "X_train_raw": X_train_raw,
            "X_val": X_val,
            "y_val": y_val,
            "X_val_seq": X_val_seq,
            "X_val_raw": X_val_raw,
            "X_test": X_test,
            "y_test": y_test,
            "X_test_seq": X_test_seq,
            "X_test_raw": X_test_raw,
            "meta_train": meta_train,
            "meta_val": meta_val,
            "meta_test": meta_test,
        }
        
    return folds


# =====================================================================
# 3. Multi-Dataset Pipeline & Serialization
# =====================================================================

def generate_all_loco_splits(
    features_df_or_path: Optional[Union[pd.DataFrame, str, Path]] = None,
    save_output: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    val_ratio: float = 0.15,
    window_size: int = 10,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Executes LOCO partitioning for both NASA Ames and CALCE benchmarks,
    saves the serialized dictionary to 'data/processed/loco_splits.pkl'.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path or (DATA_PROCESSED_DIR / "loco_splits.pkl"))
    
    if features_df_or_path is None or isinstance(features_df_or_path, (str, Path)):
        csv_path = Path(features_df_or_path or (DATA_PROCESSED_DIR / "features_all.csv"))
        if not csv_path.exists():
            from src.feature_engineering_ic import merge_all_features
            df = merge_all_features(save_output=True, verbose=verbose)
        else:
            df = pd.read_csv(csv_path)
    else:
        df = features_df_or_path
        
    if verbose:
        print("\n" + "=" * 80)
        print(" Generating Leakage-Free Leave-One-Cell-Out (LOCO) Splits")
        print("=" * 80)
        
    all_splits: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for ds in ["NASA", "CALCE"]:
        if verbose:
            print(f"\n--- Processing {ds} Benchmark Folds ---")
        folds = get_loco_splits(df, dataset_name=ds, val_ratio=val_ratio, window_size=window_size)
        all_splits[ds] = folds
        if verbose:
            for cell, f in folds.items():
                print(f"  Fold {f['fold_idx']} | Test: {cell:<8} | Train: {len(f['X_train']):>4} samples | Val: {len(f['X_val']):>4} samples | Test: {len(f['X_test']):>4} samples")
                
    if save_output:
        with open(output_path, "wb") as f:
            pickle.dump(all_splits, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"\n[OK] Saved finalized LOCO splits to {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
            
    return all_splits


# =====================================================================
# 4. Summary Table & Validation Reporting
# =====================================================================

def generate_loco_summary_table(
    all_splits: Dict[str, Dict[str, Dict[str, Any]]],
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Generates detailed EDA summary table reporting sample counts, SOH ranges,
    and feature standardization checks across all folds.
    """
    RESULTS_TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    
    for ds_name, folds in all_splits.items():
        for test_cell, f in folds.items():
            X_tr = f["X_train"]
            y_tr = f["y_train"]
            y_va = f["y_val"]
            y_te = f["y_test"]
            
            # Check training normalization mean & std
            tr_mean_err = np.max(np.abs(np.mean(X_tr, axis=0)))
            tr_std_err = np.max(np.abs(np.std(X_tr, axis=0) - 1.0))
            
            rows.append({
                "Dataset": ds_name,
                "Fold": f["fold_idx"],
                "Held-Out Test Cell": test_cell,
                "Train Cells": ", ".join(f["train_cells"]),
                "Train Samples": len(f["X_train"]),
                "Val Samples": len(f["X_val"]),
                "Test Samples": len(f["X_test"]),
                "Total Samples": len(f["X_train"]) + len(f["X_val"]) + len(f["X_test"]),
                "Train SOH Range (%)": f"{np.min(y_tr):.1f} - {np.max(y_tr):.1f}",
                "Val SOH Range (%)": f"{np.min(y_va):.1f} - {np.max(y_va):.1f}",
                "Test SOH Range (%)": f"{np.min(y_te):.1f} - {np.max(y_te):.1f}",
                "Train Scaler Mean": round(float(tr_mean_err), 4),
                "Train Scaler Std": round(float(1.0 + tr_std_err), 4),
            })
            
    df_summary = pd.DataFrame(rows)
    tab_path = RESULTS_TAB_DIR / "loco_splits_summary.csv"
    df_summary.to_csv(tab_path, index=False)
    
    if print_table:
        print("\n" + "=" * 118)
        print(" WD-GPR-SR Leave-One-Cell-Out (LOCO) Cross-Validation: Data Leakage Prevention Summary")
        print("=" * 118)
        print(f"{'Dataset':<8} {'Fold':<5} {'Test Cell':<10} {'Train N':<8} {'Val N':<6} {'Test N':<7} {'Train SOH Range':<18} {'Test SOH Range':<18} {'Scaler Check':<15}")
        print("-" * 118)
        for _, r in df_summary.iterrows():
            scaler_ok = "Passed (μ≈0, σ≈1)" if r['Train Scaler Mean'] < 1e-4 else "Check"
            print(f"{r['Dataset']:<8} {r['Fold']:<5} {r['Held-Out Test Cell']:<10} {r['Train Samples']:<8} {r['Val Samples']:<6} {r['Test Samples']:<7} {r['Train SOH Range (%)']:<18} {r['Test SOH Range (%)']:<18} {scaler_ok:<15}")
        print("=" * 118)
        
    return df_summary


# =====================================================================
# 5. Visualization: LOCO Cross-Validation Split Diagram
# =====================================================================

def plot_loco_split_diagram(
    all_splits: Dict[str, Dict[str, Dict[str, Any]]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots a publication-quality diagram illustrating the Leave-One-Cell-Out
    partitioning scheme for NASA and CALCE benchmarks.
    """
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path or (RESULTS_FIG_DIR / "loco_cv_splits_diagram.png"))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Colors
    c_train = "#2b5c8f"    # Deep Blue
    c_val = "#e69f00"      # Golden Yellow / Orange
    c_test = "#d55e00"     # Vermilion Red
    
    benchmarks = [("NASA", NASA_CELLS, axes[0]), ("CALCE", CALCE_CELLS, axes[1])]
    
    for ds_name, cell_list, ax in benchmarks:
        folds = all_splits.get(ds_name, {})
        n_folds = len(cell_list)
        
        y_positions = np.arange(n_folds)
        
        for fold_idx, test_cell in enumerate(cell_list):
            f = folds.get(test_cell)
            if f is None:
                continue
                
            y = n_folds - 1 - fold_idx
            x_offset = 0
            
            for c in cell_list:
                if c == test_cell:
                    # Test cell block
                    n_c = len(f["X_test"])
                    ax.barh(y, n_c, left=x_offset, height=0.55, color=c_test, edgecolor="black", linewidth=0.8)
                    ax.text(x_offset + n_c/2, y, f"{c}\n(Test: {n_c})", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
                    x_offset += n_c
                else:
                    # Development cell: 85% train, 15% val
                    # Find count from meta
                    meta_tr = f["meta_train"]
                    meta_va = f["meta_val"]
                    n_tr_c = len(meta_tr[meta_tr["cell_id"] == c])
                    n_va_c = len(meta_va[meta_va["cell_id"] == c])
                    
                    ax.barh(y, n_tr_c, left=x_offset, height=0.55, color=c_train, edgecolor="black", linewidth=0.8)
                    ax.text(x_offset + n_tr_c/2, y, f"{c} (Tr)", ha="center", va="center", color="white", fontsize=7.5)
                    x_offset += n_tr_c
                    
                    ax.barh(y, n_va_c, left=x_offset, height=0.55, color=c_val, edgecolor="black", linewidth=0.8)
                    ax.text(x_offset + n_va_c/2, y, "V", ha="center", va="center", color="black", fontsize=7, fontweight="bold")
                    x_offset += n_va_c
                    
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"Fold {n_folds - 1 - i} ({cell_list[n_folds - 1 - i]})" for i in range(n_folds)], fontweight="bold")
        ax.set_xlabel("Cumulative Cycle Number across Cells", fontsize=10)
        ax.set_title(f"({ds_name}) Leave-One-Cell-Out Cross-Validation Scheme", fontweight="bold", fontsize=11, pad=12)
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")
        
    # Legend
    legend_elements = [
        patches.Patch(facecolor=c_train, edgecolor="black", label="Training Partition (85% chronological per dev cell)"),
        patches.Patch(facecolor=c_val, edgecolor="black", label="Validation Partition (15% chronological tail per dev cell)"),
        patches.Patch(facecolor=c_test, edgecolor="black", label="Held-Out Test Cell (100% unseen cell)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved LOCO CV split diagram to {save_path}")


# =====================================================================
# Main Execution Runner
# =====================================================================

if __name__ == "__main__":
    splits = generate_all_loco_splits(save_output=True, verbose=True)
    summary_df = generate_loco_summary_table(splits, print_table=True)
    plot_loco_split_diagram(splits, show=False)

