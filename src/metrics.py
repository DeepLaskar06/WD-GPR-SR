"""
Shared Evaluation Metrics for WD-GPR-SR.
Author: Moonis Laskar (Roll No: 2401CS29)

Includes:
1. Accuracy Metrics: MAE, MSE, RMSE, R2, MAPE
2. Calibrated Uncertainty Metrics: NLL, PICP (95%), MPIW (95%)
3. Symbolic Complexity Metric: count_sympy_nodes
"""

from typing import Dict, Optional, Union, Any
import numpy as np
from scipy import stats
import sympy


# =====================================================================
# 1. Deterministic / Point-Prediction Accuracy Metrics
# =====================================================================

def MAE(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Mean Absolute Error."""
    y_t, y_p = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def MSE(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Mean Squared Error."""
    y_t, y_p = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_t - y_p) ** 2))


def RMSE(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(MSE(y_true, y_pred)))


def R2(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Coefficient of Determination (R-squared).
    R^2 = 1 - (SS_res / SS_tot).
    """
    y_t, y_p = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def MAPE(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (in percentage: 0% - 100%+).
    Uses epsilon in denominator to avoid division by zero.
    """
    y_t, y_p = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    denom = np.where(np.abs(y_t) < eps, eps, np.abs(y_t))
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


# =====================================================================
# 2. Calibrated Uncertainty Quantification Metrics
# =====================================================================

def NLL(
    y_true: Union[np.ndarray, list],
    y_mean: Union[np.ndarray, list],
    y_var: Union[np.ndarray, list],
    eps: float = 1e-6,
) -> float:
    """
    Gaussian Negative Log-Likelihood (NLL).
    NLL = 0.5 * mean( log(2 * pi * (var + eps)) + (y_true - mean)^2 / (var + eps) )
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(y_mean, dtype=np.float64)
    var = np.maximum(np.asarray(y_var, dtype=np.float64), eps)
    
    nll_vals = 0.5 * (np.log(2.0 * np.pi * var) + ((y_t - mu) ** 2) / var)
    return float(np.mean(nll_vals))


def PICP(
    y_true: Union[np.ndarray, list],
    y_mean: Union[np.ndarray, list],
    y_var: Union[np.ndarray, list],
    confidence: float = 0.95,
) -> float:
    """
    Prediction Interval Coverage Probability (PICP).
    Computes fraction of true values that fall inside the (1 - alpha) prediction interval.
    Nominal coverage for 0.95 confidence is >= 95.0%.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(y_mean, dtype=np.float64)
    sigma = np.sqrt(np.maximum(np.asarray(y_var, dtype=np.float64), 1e-12))
    
    alpha = 1.0 - confidence
    z_score = stats.norm.ppf(1.0 - alpha / 2.0)
    
    lower_bound = mu - z_score * sigma
    upper_bound = mu + z_score * sigma
    
    inside = (y_t >= lower_bound) & (y_t <= upper_bound)
    return float(np.mean(inside) * 100.0)


def MPIW(
    y_mean: Union[np.ndarray, list],
    y_var: Union[np.ndarray, list],
    confidence: float = 0.95,
) -> float:
    """
    Mean Prediction Interval Width (MPIW).
    Measures the average width of the (1 - alpha) confidence interval.
    Smaller MPIW with high PICP indicates a sharper, well-calibrated distribution.
    """
    sigma = np.sqrt(np.maximum(np.asarray(y_var, dtype=np.float64), 1e-12))
    alpha = 1.0 - confidence
    z_score = stats.norm.ppf(1.0 - alpha / 2.0)
    
    width = 2.0 * z_score * sigma
    return float(np.mean(width))


# =====================================================================
# 3. Complexity Metric for Discovered Mathematical Expressions
# =====================================================================

def count_sympy_nodes(expr: Union[sympy.Expr, str, tuple, Any]) -> int:
    """
    Counts the total number of operations, functions, variables, and constants
    in a SymPy expression tree (Occam's razor complexity).
    Handles functional grammar (gplearn add, sub, mul, div) and protects against
    tuple returns from polynomial division.
    """
    if expr is None:
        return 0
        
    if isinstance(expr, tuple):
        return sum(count_sympy_nodes(sub_e) for sub_e in expr)
        
    if isinstance(expr, str):
        locals_dict = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x / (y + 1e-12),
            "sin": sympy.sin,
            "cos": sympy.cos,
            "exp": sympy.exp,
            "log": lambda x: sympy.log(abs(x) + 1e-12),
            "sqrt": lambda x: sympy.sqrt(abs(x)),
            "abs": abs,
            "neg": lambda x: -x,
            "inv": lambda x: 1.0 / (x + 1e-12),
        }
        for i in range(100):
            locals_dict[f"X{i}"] = sympy.Symbol(f"X{i}")
            locals_dict[f"x{i}"] = sympy.Symbol(f"x{i}")
            
        try:
            expr = sympy.sympify(expr, locals=locals_dict)
        except Exception:
            import re
            tokens = re.findall(r"[A-Za-z0-9_.]+|[+\-*/()]", str(expr))
            return max(1, len(tokens))
            
    if isinstance(expr, tuple):
        return sum(count_sympy_nodes(sub_e) for sub_e in expr)
        
    if isinstance(expr, (sympy.Basic, sympy.Expr)):
        return sum(1 for _ in sympy.preorder_traversal(expr))
        
    return 1


# =====================================================================
# Unified Evaluator
# =====================================================================

def evaluate_predictions(
    y_true: Union[np.ndarray, list],
    y_mean: Union[np.ndarray, list],
    y_var: Optional[Union[np.ndarray, list]] = None,
    expr: Optional[Union[sympy.Expr, str]] = None,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Computes all 9 evaluation metrics simultaneously.
    """
    res = {
        "MAE": MAE(y_true, y_mean),
        "MSE": MSE(y_true, y_mean),
        "RMSE": RMSE(y_true, y_mean),
        "R2": R2(y_true, y_mean),
        "MAPE": MAPE(y_true, y_mean),
    }
    
    if y_var is not None:
        res["NLL"] = NLL(y_true, y_mean, y_var)
        res[f"PICP_{int(confidence*100)}"] = PICP(y_true, y_mean, y_var, confidence=confidence)
        res[f"MPIW_{int(confidence*100)}"] = MPIW(y_mean, y_var, confidence=confidence)
        
    if expr is not None:
        res["Node_Count"] = float(count_sympy_nodes(expr))
        
    return res

