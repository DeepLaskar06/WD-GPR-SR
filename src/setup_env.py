"""
Environment Verification, Hardware Diagnostics, and Unified Symbolic Regression Wrapper.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction
"""

import os
import sys
import io
import contextlib
import importlib
import importlib.metadata
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

# Suppress Matplotlib writable directory and PyTorch deprecation notices
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(BASE_DIR / ".julia"))
os.environ.setdefault("PYTHON_JULIAPKG_DIR", str(BASE_DIR / ".julia"))
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
# 1. Package Verification & Hardware Diagnostics
# =====================================================================

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("gpytorch", "gpytorch"),
    ("xgboost", "xgboost"),
    ("pysr", "pysr"),
    ("gplearn", "gplearn"),
    ("sympy", "sympy"),
    ("shap", "shap"),
    ("lime", "lime"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("tqdm", "tqdm"),
    ("requests", "requests"),
]

def check_package_status() -> Dict[str, Tuple[bool, str]]:
    """
    Checks import status and package versions of all required project dependencies.
    Safe against runtime initialization errors (such as Julia runtime checks in PySR).
    """
    results = {}
    for import_name, display_name in REQUIRED_PACKAGES:
        # 1. Query distribution metadata first
        dist_version = None
        for name in (display_name, import_name, display_name.replace("-", "_")):
            try:
                dist_version = importlib.metadata.version(name)
                if dist_version:
                    break
            except Exception:
                continue

        # 2. Test importability with safe handling for backend-dependent packages
        try:
            if import_name == "pysr":
                if dist_version:
                    results[display_name] = (True, f"{dist_version} (Python package ready; Julia runtime pending)")
                else:
                    results[display_name] = (False, "Not installed")
                continue

            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", dist_version or "installed")
            results[display_name] = (True, str(version))
        except Exception as e:
            if dist_version:
                results[display_name] = (True, f"{dist_version} (installed; import notice: {type(e).__name__})")
            else:
                results[display_name] = (False, f"Not installed ({type(e).__name__})")
                
    return results


def check_hardware_and_compute() -> Dict[str, Any]:
    """
    Detects CPU architecture, platform, and CUDA / GPU acceleration availability.
    """
    info = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "cuda_available": False,
        "device_count": 0,
        "device_name": "CPU",
        "pytorch_version": "N/A",
    }
    
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
        
    return info


def check_symbolic_backends() -> Dict[str, Any]:
    """
    Checks availability of PySR (Julia backend) and gplearn.
    """
    status = {
        "pysr_available": False,
        "pysr_version": None,
        "pysr_julia_ready": False,
        "gplearn_available": False,
        "gplearn_version": None,
        "active_backend": None,
    }
    
    # Check PySR package availability
    try:
        pysr_ver = importlib.metadata.version("pysr")
        status["pysr_available"] = True
        status["pysr_version"] = str(pysr_ver)
    except Exception:
        status["pysr_available"] = False

    # Safely probe Julia readiness without crashing on network or permission errors
    if status["pysr_available"]:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                import pysr
                if hasattr(pysr, "julia_helpers") and hasattr(pysr.julia_helpers, "is_julia_available"):
                    status["pysr_julia_ready"] = bool(pysr.julia_helpers.is_julia_available())
                else:
                    status["pysr_julia_ready"] = False
        except Exception:
            status["pysr_julia_ready"] = False

    # Check gplearn
    try:
        import gplearn
        gplearn_ver = getattr(gplearn, "__version__", None)
        if not gplearn_ver:
            gplearn_ver = importlib.metadata.version("gplearn")
        status["gplearn_available"] = True
        status["gplearn_version"] = str(gplearn_ver)
    except Exception:
        status["gplearn_available"] = False
        
    # Determine default active backend
    if status["pysr_available"] and status["pysr_julia_ready"]:
        status["active_backend"] = "PySR"
    elif status["gplearn_available"]:
        status["active_backend"] = "gplearn"
    else:
        status["active_backend"] = "heuristic_fallback"
        
    return status


# =====================================================================
# 2. Unified Uncertainty-Weighted Symbolic Regression Wrapper
# =====================================================================

class UncertaintyWeightedSR:
    """
    Unified Uncertainty-Weighted Symbolic Regressor.
    
    Supports:
      1. PySRRegressor: passing weights = 1.0 / (sigma_sq + 1e-6)
      2. gplearn.genetic.SymbolicRegressor: passing sample_weight = 1.0 / (sigma_sq + 1e-6)
      3. Built-in polynomial/ridge regression fallback if neither engine is available.
      
    Mathematical Objective:
      L_SR(g) = (1 / N) * sum_i [ (r_i - g(x_i))^2 / (sigma_WD^2(x_i) + eps) ] + lambda * Nodes(g)
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        population_size: int = 1000,
        generations: int = 20,
        parsimony_coefficient: float = 0.005,
        random_state: int = 42,
        eps: float = 1e-6,
        **kwargs
    ):
        self.requested_backend = backend
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state
        self.eps = eps
        self.kwargs = kwargs
        
        self.backend_used: Optional[str] = None
        self.model = None
        self.discovered_equation = None
        self.feature_names = None
        
        self._init_backend()

    def _init_backend(self):
        backend_status = check_symbolic_backends()
        target = self.requested_backend or backend_status["active_backend"]
        
        # 1. Try PySR if requested and Julia runtime is verified
        if target == "PySR" and backend_status["pysr_available"] and backend_status["pysr_julia_ready"]:
            try:
                from pysr import PySRRegressor
                self.model = PySRRegressor(
                    niterations=self.generations,
                    populations=15,
                    population_size=min(self.population_size, 50),
                    parsimony=self.parsimony_coefficient,
                    random_state=self.random_state,
                    verbosity=0,
                    **self.kwargs
                )
                self.backend_used = "PySR"
                return
            except Exception as e:
                warnings.warn(f"Failed to initialize PySR: {e}. Falling back to gplearn.")
                
        # 2. Try gplearn (pure Python genetic programming)
        if (target in ("gplearn", "PySR")) and backend_status["gplearn_available"]:
            try:
                from gplearn.genetic import SymbolicRegressor
                self.model = SymbolicRegressor(
                    population_size=self.population_size,
                    generations=self.generations,
                    parsimony_coefficient=self.parsimony_coefficient,
                    random_state=self.random_state,
                    p_crossover=0.7,
                    p_subtree_mutation=0.1,
                    p_hoist_mutation=0.05,
                    p_point_mutation=0.1,
                    max_samples=0.9,
                    verbose=0,
                    **self.kwargs
                )
                self.backend_used = "gplearn"
                return
            except Exception as e:
                warnings.warn(f"Failed to initialize gplearn: {e}. Falling back to heuristic baseline.")

        # 3. Deterministic Heuristic Fallback
        self.backend_used = "heuristic_fallback"
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sigma_sq: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ):
        """
        Fits symbolic regression to residuals y using uncertainty weighting:
          w_i = 1.0 / (sigma_sq_i + eps)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]

        # Compute heteroscedastic weights: w_i = 1 / (sigma^2 + eps)
        if sigma_sq is None:
            weights = np.ones_like(y, dtype=np.float64)
        else:
            sigma_sq = np.asarray(sigma_sq, dtype=np.float64).flatten()
            weights = 1.0 / (np.maximum(sigma_sq, 0.0) + self.eps)
            
        # Normalize weights to mean 1.0 for numerical balance
        weights = weights / (np.mean(weights) + 1e-12)

        if self.backend_used == "PySR":
            self.model.fit(X, y, weights=weights, variable_names=self.feature_names)
            try:
                self.discovered_equation = str(self.model.sympy())
            except Exception:
                self.discovered_equation = str(self.model.get_best()["equation"])
        elif self.backend_used == "gplearn":
            self.model.fit(X, y, sample_weight=weights)
            self.discovered_equation = str(self.model._program)
        else:
            # Fallback: Weighted Ridge / Least-Squares 2nd degree polynomial
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            clf = Ridge(alpha=1.0, random_state=self.random_state)
            clf.fit(X_poly, y, sample_weight=weights)
            
            self._fallback_poly = poly
            self._fallback_clf = clf
            
            feature_names_poly = poly.get_feature_names_out(self.feature_names)
            terms = [f"{coef:+.4f}*{name}" for coef, name in zip(clf.coef_, feature_names_poly) if abs(coef) > 1e-4]
            intercept = f"{clf.intercept_:+.4f}"
            self.discovered_equation = " + ".join([intercept] + terms) if terms else intercept

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts residual correction for input feature matrix X."""
        X = np.asarray(X, dtype=np.float64)
        if self.backend_used in ("PySR", "gplearn"):
            return self.model.predict(X)
        else:
            X_poly = self._fallback_poly.transform(X)
            return self._fallback_clf.predict(X_poly)

    def get_equation(self) -> str:
        """Returns string representation of the discovered symbolic equation."""
        return self.discovered_equation or "0.0"

    def to_sympy(self):
        """Converts discovered equation into a valid SymPy analytical expression."""
        import sympy
        eq_str = self.get_equation()
        
        if self.backend_used == "gplearn":
            # Map gplearn functional grammar to SymPy mathematical operators
            locals_dict = {
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
                "mul": lambda a, b: a * b,
                "div": lambda a, b: a / (b + 1e-12),
                "neg": lambda a: -a,
                "inv": lambda a: 1.0 / (a + 1e-12),
                "abs": sympy.Abs,
                "sqrt": sympy.sqrt,
                "log": sympy.log,
                "sin": sympy.sin,
                "cos": sympy.cos,
            }
            if self.feature_names:
                for i, name in enumerate(self.feature_names):
                    locals_dict[f"X{i}"] = sympy.Symbol(name)
                    locals_dict[name] = sympy.Symbol(name)
            try:
                return sympy.sympify(eq_str, locals=locals_dict)
            except Exception:
                pass
                
        try:
            return sympy.sympify(eq_str)
        except Exception:
            return sympy.sympify("0.0")


# =====================================================================
# 3. Diagnostic Runner
# =====================================================================

def run_diagnostics(verbose: bool = True) -> Dict[str, Any]:
    """
    Executes full environment verification, prints report, and returns status dict.
    """
    pkg_status = check_package_status()
    hw_info = check_hardware_and_compute()
    sr_status = check_symbolic_backends()
    
    if verbose:
        print("=" * 70)
        print(" WD-GPR-SR Environment & Hardware Verification (Day 2)")
        print(" Author: Moonis Laskar | Roll No: 2401CS29")
        print("=" * 70)
        
        print("\n[1] Hardware & Runtime:")
        print(f"  * Python Version  : {hw_info['python_version']} ({hw_info['platform']})")
        print(f"  * PyTorch Version : {hw_info['pytorch_version']}")
        print(f"  * CUDA Available  : {hw_info['cuda_available']}")
        if hw_info['cuda_available']:
            print(f"  * GPU Device      : {hw_info['device_name']} (Count: {hw_info['device_count']})")
            print(f"  * CUDA Version    : {hw_info.get('cuda_version', 'N/A')}")
        else:
            print("  * Compute Device  : CPU (Intel/AMD/ARM)")

        print("\n[2] Required Packages Status:")
        for pkg, (installed, ver) in pkg_status.items():
            status_symbol = "[OK]" if installed else "[MISSING]"
            print(f"  {status_symbol:<10} {pkg:<15} : {ver}")

        print("\n[3] Symbolic Regression Backends:")
        print(f"  * PySR Available  : {sr_status['pysr_available']} (Julia Ready: {sr_status['pysr_julia_ready']})")
        print(f"  * gplearn Available: {sr_status['gplearn_available']} (Version: {sr_status['gplearn_version']})")
        print(f"  * Active SR Mode  : {sr_status['active_backend']}")
        print("=" * 70)
        
    return {
        "packages": pkg_status,
        "hardware": hw_info,
        "symbolic": sr_status,
    }


if __name__ == "__main__":
    run_diagnostics(verbose=True)
