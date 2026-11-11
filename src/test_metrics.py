"""
Unit tests for WD-GPR-SR evaluation metrics.
Author: Moonis Laskar (Roll No: 2401CS29)
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sympy
from src.metrics import (
    MAE, MSE, RMSE, R2, MAPE,
    NLL, PICP, MPIW,
    count_sympy_nodes,
    evaluate_predictions
)

def run_metric_unit_tests():
    print("=" * 60)
    print(" Running WD-GPR-SR Metrics Unit Test Suite (9 Metrics)")
    print("=" * 60)
    
    # Generate synthetic dummy test data
    np.random.seed(42)
    N = 100
    y_true = np.linspace(1.0, 0.7, N)  # degrading SOH trajectory from 1.0 to 0.7
    noise = np.random.normal(0, 0.01, N)
    y_pred = y_true + noise
    y_var = np.full(N, 0.01 ** 2)      # known variance (0.01 std)
    
    # 1. Deterministic Metrics
    mae_val = MAE(y_true, y_pred)
    mse_val = MSE(y_true, y_pred)
    rmse_val = RMSE(y_true, y_pred)
    r2_val = R2(y_true, y_pred)
    mape_val = MAPE(y_true, y_pred)
    
    print("\n[Accuracy Metrics on Synthetic Degradation]:")
    print(f"  * MAE  : {mae_val:.6f}")
    print(f"  * MSE  : {mse_val:.6f}")
    print(f"  * RMSE : {rmse_val:.6f}")
    print(f"  * R2   : {r2_val:.6f}")
    print(f"  * MAPE : {mape_val:.4f}%")
    
    assert mae_val > 0.0 and mae_val < 0.05, f"MAE out of expected bounds: {mae_val}"
    assert rmse_val > 0.0 and rmse_val < 0.05, f"RMSE out of expected bounds: {rmse_val}"
    assert r2_val > 0.90, f"R2 unexpectedly low: {r2_val}"
    
    # 2. Uncertainty Metrics
    nll_val = NLL(y_true, y_pred, y_var)
    picp_val = PICP(y_true, y_pred, y_var, confidence=0.95)
    mpiw_val = MPIW(y_pred, y_var, confidence=0.95)
    
    print("\n[Uncertainty Calibration Metrics (95% Interval)]:")
    print(f"  * NLL  : {nll_val:.4f}")
    print(f"  * PICP : {picp_val:.2f}% (Expected ~95.0% for true Gaussian noise)")
    print(f"  * MPIW : {mpiw_val:.6f} (Expected ~2 * 1.96 * 0.01 = 0.0392)")
    
    assert 90.0 <= picp_val <= 100.0, f"PICP not calibrated: {picp_val}"
    assert abs(mpiw_val - 0.0392) < 0.005, f"MPIW deviates from theoretical value: {mpiw_val}"
    
    # 3. Complexity Metric (SymPy Expression Tree Nodes)
    x = sympy.Symbol('x')
    test_expr = 0.05 * sympy.sin(x) + sympy.exp(-0.1 * x)
    node_count = count_sympy_nodes(test_expr)
    
    print("\n[Symbolic Complexity Metric]:")
    print(f"  * Expression: {test_expr}")
    print(f"  * SymPy Tree Node Count: {node_count}")
    assert node_count > 0, "Node count should be positive"
    
    # 4. Unified Batch Evaluation
    summary = evaluate_predictions(y_true, y_pred, y_var, expr=test_expr)
    print("\n[Unified Evaluator Output Dictionary]:")
    for k, v in summary.items():
        print(f"  {k:<10}: {v:.6f}")
        
    print("\n" + "=" * 60)
    print(" [OK] All 9 Metrics Unit Tests PASSED Successfully!")
    print("=" * 60)
    return summary

if __name__ == "__main__":
    run_metric_unit_tests()
