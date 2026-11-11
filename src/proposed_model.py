"""
Proposed Model: Wide-and-Deep Sparse Gaussian Process with Uncertainty-Weighted Symbolic Residual Correction (WD-GPR-SR).
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

Implements the 3-stage proposed framework:
  Stage 1 (Days 13-14): Wide-and-Deep Sparse GPR (Variational ELBO, M=64 inducing points)
  Stage 2 (Days 15-16): Uncertainty-Weighted Symbolic Residual Regression (Weights = 1 / (sigma^2 + 1e-6))
  Stage 3 (Days 17-18): Full Pipeline Integration & Calibrated Closed-Form Inference
"""

import os
import sys
import time
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
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

from src.config import set_global_seed, PRIMARY_SEED
from src.metrics import evaluate_predictions, count_sympy_nodes
from src.setup_env import UncertaintyWeightedSR


# =====================================================================
# Stage 1: Wide-and-Deep Sparse GPR Components (Days 13-14)
# =====================================================================

class DeepFeatureExtractor(nn.Module):
    """
    Deep Neural Feature Extractor phi_theta(x): R^D -> R^32 -> R^16 -> R^8.
    Learns a deterministic non-linear warp of raw battery health indicators
    to capture complex electrochemical phase transitions and knee-point regimes.
    """
    def __init__(
        self,
        input_dim: int = 10,
        hidden1: int = 32,
        hidden2: int = 16,
        out_dim: int = 8,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WDCovariance(gpytorch.kernels.Kernel):
    """
    Wide-and-Deep Covariance Kernel:
      k_WD(x, x') = k_Wide(x, x') + k_Deep(phi(x), phi(x')) + k_stabilizer(x, x')
      
    Branches:
      1. Wide Branch: ScaleKernel(LinearKernel(ard_num_dims=D))
         Captures the monotonic, macroscopic SOH degradation trend.
      2. Deep Branch: ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8))
         Operates on phi_theta(x) to capture localized knee points and regeneration humps.
      3. Stabilizer Branch: ScaleKernel(RBFKernel())
         Provides full-rank support to ensure strict positive-definiteness of K_MM for M inducing points.
    """
    def __init__(
        self,
        feature_extractor: nn.Module,
        input_dim: int = 10,
        deep_dim: int = 8,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.wide_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(ard_num_dims=input_dim)
        )
        self.deep_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=deep_dim)
        )
        self.stabilizer_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.stabilizer_kernel.outputscale = 1e-3

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        k_wide = self.wide_kernel(x1, x2, diag=diag, **params)
        
        phi_x1 = self.feature_extractor(x1)
        phi_x2 = self.feature_extractor(x2) if (x2 is not x1 and not diag) else phi_x1
        k_deep = self.deep_kernel(phi_x1, phi_x2, diag=diag, **params)
        
        k_stab = self.stabilizer_kernel(x1, x2, diag=diag, **params)
        return k_wide + k_deep + k_stab


class WD_SparseGPR(gpytorch.models.ApproximateGP):
    """
    Wide-and-Deep Sparse Variational Gaussian Process Regressor.
    Utilizes M inducing points and optimizes the Variational ELBO.
    Initializes ConstantMean to the empirical target mean for rapid convergence.
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
        self.covar_module = WDCovariance(
            self.feature_extractor,
            input_dim=input_dim,
            deep_dim=deep_dim,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        # Pass x as single argument to preserve exact symmetry flag in LinearOperator
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_wd_sparse_gpr(
    model: WD_SparseGPR,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    verbose: bool = False,
) -> Tuple[WD_SparseGPR, gpytorch.likelihoods.GaussianLikelihood, List[float]]:
    """
    Trains the Wide-and-Deep Sparse GPR by maximizing the Variational ELBO.
    """
    t_x = torch.tensor(X_train, dtype=torch.float32)
    t_y = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(t_x, t_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=t_y.size(0))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    losses = []
    with gpytorch.settings.cholesky_jitter(1e-3):
        model.train()
        likelihood.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for bx, by in loader:
                optimizer.zero_grad()
                output = model(bx)
                loss = -mll(output, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / max(1, n_batches)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    [WD-SparseGPR] Epoch {epoch+1:02d}/{epochs:02d} | ELBO Loss: {avg_loss:.4f}", flush=True)
                
    return model, likelihood, losses


# =====================================================================
# Stage 2: Uncertainty-Weighted Symbolic Residual Regression (Days 15-16)
# =====================================================================

def fit_residual_sr(
    wd_gpr_model: WD_SparseGPR,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    generations: int = 10,
    population_size: int = 200,
    feature_names: Optional[List[str]] = None,
) -> Tuple[UncertaintyWeightedSR, str, np.ndarray, np.ndarray]:
    """
    Fits Uncertainty-Weighted Symbolic Regression on the GPR residuals:
      r_i = y_i - mu_WD(x_i)
      w_i = 1.0 / (sigma_WD^2(x_i) + 1e-6)
      
    Downweights high-uncertainty regions (e.g. noise / anomalous spikes),
    allowing genetic programming to discover parsimonious closed-form equations.
    """
    set_global_seed(seed)
    
    wd_gpr_model.eval()
    likelihood.eval()
    
    t_x = torch.tensor(X_train, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
        pred_dist = likelihood(wd_gpr_model(t_x))
        mu_train = pred_dist.mean.cpu().numpy()
        var_train = pred_dist.variance.cpu().numpy()
        
    residuals = y_train - mu_train
    
    # Initialize unified symbolic regression wrapper
    sr = UncertaintyWeightedSR(
        population_size=population_size,
        generations=generations,
        random_state=seed,
        parsimony_coefficient=0.005,
    )
    
    # Fit with heteroscedastic weights: w_i = 1.0 / (sigma_i^2 + 1e-6)
    sr.fit(X_train, residuals, sigma_sq=var_train, feature_names=feature_names)
    best_equation = sr.get_equation()
    
    return sr, best_equation, residuals, var_train


# =====================================================================
# Stage 3: Full Pipeline Integration & Inference Wrapper (Days 17-18)
# =====================================================================

class WD_GPR_SR_Pipeline:
    """
    Full WD-GPR-SR Prognostic Pipeline.
    
    Predicts:
      y_pred(x) = mu_WD(x) + g_SR(x)
      y_var(x)  = sigma_WD^2(x)
    """
    def __init__(
        self,
        input_dim: int = 10,
        deep_dim: int = 8,
        num_inducing: int = 64,
        epochs_gpr: int = 35,
        batch_size: int = 64,
        lr_gpr: float = 0.03,
        generations_sr: int = 10,
        population_size_sr: int = 200,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.deep_dim = deep_dim
        self.num_inducing = num_inducing
        self.epochs_gpr = epochs_gpr
        self.batch_size = batch_size
        self.lr_gpr = lr_gpr
        self.generations_sr = generations_sr
        self.population_size_sr = population_size_sr
        self.random_state = random_state
        
        self.gpr_model: Optional[WD_SparseGPR] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self.sr_model: Optional[UncertaintyWeightedSR] = None
        self.discovered_equation: Optional[str] = None
        self.training_losses: List[float] = []
        self.residual_std: float = 1.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> "WD_GPR_SR_Pipeline":
        """
        Executes Stage 1 (WD-SparseGPR) followed by Stage 2 (Residual SR).
        """
        set_global_seed(self.random_state)
        
        # 1. Inducing points via K-Means
        M = min(self.num_inducing, len(X_train))
        kmeans = KMeans(n_clusters=M, random_state=self.random_state, n_init=1).fit(X_train)
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        # 2. Initialize Stage 1 WD-SparseGPR with empirical mean initialization
        y_mean = float(y_train.mean())
        self.gpr_model = WD_SparseGPR(
            inducing_points=inducing_points,
            input_dim=self.input_dim,
            deep_dim=self.deep_dim,
            y_mean=y_mean,
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Train Stage 1
        t0 = time.time()
        self.gpr_model, self.likelihood, self.training_losses = train_wd_sparse_gpr(
            self.gpr_model,
            self.likelihood,
            X_train,
            y_train,
            epochs=self.epochs_gpr,
            batch_size=self.batch_size,
            lr=self.lr_gpr,
            verbose=verbose,
        )
        if verbose:
            print(f"  [OK] Stage 1 (WD-SparseGPR) converged in {time.time()-t0:.1f}s", flush=True)
            
        # 3. Train Stage 2 (Uncertainty-Weighted Symbolic Residual Regression)
        t1 = time.time()
        self.sr_model, self.discovered_equation, residuals, _ = fit_residual_sr(
            self.gpr_model,
            self.likelihood,
            X_train,
            y_train,
            seed=self.random_state,
            generations=self.generations_sr,
            population_size=self.population_size_sr,
            feature_names=feature_names,
        )
        self.residual_std = float(residuals.std())
        if verbose:
            print(f"  [OK] Stage 2 (Uncertainty-Weighted SR) discovered: {self.discovered_equation} ({time.time()-t1:.1f}s)", flush=True)
            
        return self

    def predict(
        self,
        X_test: np.ndarray,
        return_decomposed: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Inference:
          y_pred = mu_WD(x) + g_SR(x)
          y_var  = sigma_WD^2(x)
        """
        if self.gpr_model is None or self.likelihood is None or self.sr_model is None:
            raise RuntimeError("Pipeline must be fitted before predict() is called.")
            
        self.gpr_model.eval()
        self.likelihood.eval()
        
        t_x = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
            pred_dist = self.likelihood(self.gpr_model(t_x))
            mu_wd = pred_dist.mean.cpu().numpy()
            var_wd = pred_dist.variance.cpu().numpy()
            
        res_sr = self.sr_model.predict(X_test)
        
        # Physically bound residual correction to prevent rational operator divergence
        max_dev = 3.0 * max(self.residual_std, 0.5)
        res_sr = np.clip(res_sr, -max_dev, max_dev)
        
        y_pred = mu_wd + res_sr
        
        if return_decomposed:
            return y_pred, var_wd, mu_wd, res_sr
        return y_pred, var_wd

    def get_best_equation(self) -> str:
        """Returns the discovered closed-form residual equation."""
        return self.discovered_equation or "0.0"
