"""
Baseline Model Architectures for Battery SOH Prediction.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

Implements 12 competitive baseline models across 4 families:
  Family 1 (Classical & Tree-Based): SVR, Random Forest, XGBoost
  Family 2 (Symbolic Regression)   : Standalone Symbolic Regressor (PySR / gplearn)
  Family 3 (Gaussian Processes)    : Exact GPR (RBF), Exact GPR (Matérn), Exact GPR (ARD),
                                     Sparse Variational GPR (SVGP), Deep-Kernel GPR (DKL)
  Family 4 (Deep & Recurrent NNs)  : WideNN, DeepNN, Wide & Deep NN, BiLSTM (Sequential)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

from src.config import set_global_seed
from src.metrics import evaluate_predictions
from src.setup_env import UncertaintyWeightedSR


# =====================================================================
# 1. Classical, Tree-Based, and Symbolic Baselines (Days 8–9)
# =====================================================================

def train_svr_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    C: float = 100.0,
    gamma: str = "scale",
) -> Dict[str, Any]:
    """Support Vector Regression (SVR) with RBF kernel."""
    set_global_seed(seed)
    model = SVR(kernel="rbf", C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    return {"model_name": "SVR", "metrics": metrics, "y_pred": y_pred, "y_var": None}


def train_rf_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_estimators: int = 100,
) -> Dict[str, Any]:
    """Random Forest Regressor."""
    set_global_seed(seed)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    return {"model_name": "Random Forest", "metrics": metrics, "y_pred": y_pred, "y_var": None}


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
) -> Dict[str, Any]:
    """XGBoost Gradient Boosted Decision Trees."""
    set_global_seed(seed)
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    return {"model_name": "XGBoost", "metrics": metrics, "y_pred": y_pred, "y_var": None}


def train_standalone_sr_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    generations: int = 12,
    population_size: int = 350,
) -> Dict[str, Any]:
    """Standalone Symbolic Regression (PySR / gplearn) with standard mathematical primitives."""
    set_global_seed(seed)
    sr = UncertaintyWeightedSR(
        population_size=population_size,
        generations=generations,
        random_state=seed,
        parsimony_coefficient=0.005,
    )
    # Fit without uncertainty weighting (pure standalone symbolic regression)
    sr.fit(X_train, y_train, sigma_sq=None)
    y_pred = sr.predict(X_test)
    expr_str = sr.get_equation()
    metrics = evaluate_predictions(y_test, y_pred, expr=expr_str)
    return {
        "model_name": "Standalone SR",
        "metrics": metrics,
        "y_pred": y_pred,
        "y_var": None,
        "expression": expr_str,
    }


# =====================================================================
# 2. Gaussian Process Regression Baselines (Day 10)
# =====================================================================

class ExactGPModel(gpytorch.models.ExactGP):
    """Modular Exact Gaussian Process Model supporting RBF, Matérn, and ARD kernels."""
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel_type: str = "rbf",
        ard_num_dims: Optional[int] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        if kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel_type == "ard":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims or train_x.shape[1])
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")
            
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLFeatureExtractor(nn.Module):
    """Deep feature extractor MLP: D -> 32 -> 16 -> 8."""
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DKLGPModel(gpytorch.models.ExactGP):
    """Deep Kernel Learning GP: MLP representation mapped to an RBF kernel."""
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        feature_extractor: nn.Module,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        phi_x = self.feature_extractor(x)
        mean_x = self.mean_module(phi_x)
        covar_x = self.covar_module(phi_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPModel(gpytorch.models.ApproximateGP):
    """Sparse Variational GP with M inducing points and Cholesky variational distribution."""
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gp_type: str = "rbf",
    seed: int = 42,
    num_inducing: int = 64,
    num_epochs: Optional[int] = None,
    lr: float = 0.08,
) -> Dict[str, Any]:
    """
    Unified training and evaluation harness for Exact and Approximate GP baselines.
    """
    set_global_seed(seed)
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    epochs = num_epochs or (35 if len(X_train) > 1000 else 50)
    
    if gp_type in ["rbf", "matern", "ard"]:
        model = ExactGPModel(
            train_x, train_y, likelihood,
            kernel_type=gp_type,
            ard_num_dims=X_train.shape[1] if gp_type == "ard" else None
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        likelihood.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
    elif gp_type == "sparse":
        # Select M inducing points via K-Means clustering
        kmeans = KMeans(n_clusters=min(num_inducing, len(X_train)), random_state=seed, n_init=1)
        kmeans.fit(X_train)
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        model = SparseGPModel(inducing_points)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
        
        dataset = TensorDataset(train_x, train_y)
        batch_size = 128 if len(X_train) > 1000 else 64
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        likelihood.train()
        for _ in range(epochs):
            for x_b, y_b in loader:
                optimizer.zero_grad()
                output = model(x_b)
                loss = -mll(output, y_b)
                loss.backward()
                optimizer.step()
                
    elif gp_type == "dkl":
        fe = DKLFeatureExtractor(input_dim=X_train.shape[1])
        model = DKLGPModel(train_x, train_y, likelihood, fe)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        
        model.train()
        likelihood.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
    else:
        raise ValueError(f"Unknown gp_type: {gp_type}")
        
    # Evaluation with calibrated posterior predictive distribution
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = likelihood(model(test_x))
        y_mean = dist.mean.cpu().numpy()
        y_var = dist.variance.cpu().numpy()
        
    name_map = {
        "rbf": "Exact GPR (RBF)",
        "matern": "Exact GPR (Matérn)",
        "ard": "Exact GPR (ARD)",
        "sparse": "Sparse Variational GPR",
        "dkl": "Deep-Kernel GPR",
    }
    metrics = evaluate_predictions(y_test, y_mean, y_var=y_var)
    return {
        "model_name": name_map[gp_type],
        "metrics": metrics,
        "y_pred": y_mean,
        "y_var": y_var,
    }


# =====================================================================
# 3. Neural Networks & Sequence Baselines (Days 11–12)
# =====================================================================

class WideNN(nn.Module):
    """Wide neural network with one wide hidden layer (D -> 128 -> 1)."""
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepNN(nn.Module):
    """Deep, narrow MLP (D -> 64 -> 32 -> 16 -> 1)."""
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WideAndDeepNN(nn.Module):
    """Wide & Deep NN: Linear skip connection combined with deep MLP."""
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.wide(x) + self.deep(x)).squeeze(-1)


class BiLSTMRegressor(nn.Module):
    """Bidirectional LSTM sequence model with linear regression head on final hidden state."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Extract last time-step representation
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)


def train_pytorch_nn(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """Generic training harness for deterministic PyTorch regression networks."""
    set_global_seed(seed)
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        y_pred = model(test_x).cpu().numpy()
        
    metrics = evaluate_predictions(y_test, y_pred)
    return {"model_name": model_name, "metrics": metrics, "y_pred": y_pred, "y_var": None}


def train_bilstm_baseline(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray,
    seed: int = 42,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """Trains the 3D sequential BiLSTM model."""
    set_global_seed(seed)
    model = BiLSTMRegressor(input_dim=X_train_seq.shape[2], hidden_dim=32)
    return train_pytorch_nn(
        model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        model_name="BiLSTM", seed=seed, epochs=epochs, batch_size=batch_size, lr=lr
    )

