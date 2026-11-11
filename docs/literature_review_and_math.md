# Literature Review & Mathematical Formulation
**Author:** Moonis Laskar | **Roll No:** 2401CS29  
**Project:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

---

## 1. Literature Review & Research Gap Analysis

### 1.1 Comparative Methodology Table

| Method Category | Representative Approach | Accuracy on Non-Linear Knees | Calibrated Uncertainty | Closed-Form Interpretability | Noise Robustness | Key Limitation |
|---|---|---|---|---|---|---|
| **Standard GPR for Battery SOH** | Saha & Goebel (2009) — RUL prediction via standard squared-exponential GPR on NASA PCOE data | Moderate (RMSE ~2–4% SOH) | ✅ Full posterior | ❌ Kernel, not equation | Moderate (Gaussian noise only) | $O(N^3)$ training; collapses near capacity-knee; single homoscedastic noise |
| **Sparse GPR (SGPR/SVGP)** | Titsias (2009) SVGP; Liu et al. (2019) — sparse GPR for Li-ion aging with inducing points | Moderate-High | ✅ Approximate posterior | ❌ | High (inducing points filter noise) | Inducing-point placement is non-trivial; kernel mismatch at non-linearities |
| **ARD / Multi-kernel GPR** | Richardson et al. (2019) — ARD-RBF kernels; Tagade et al. (2020) — composite kernels for fast/slow degradation modes | High on monotone regions | ✅ | ❌ | Moderate | Cannot learn feature representations; manually engineered kernel sums |
| **Deep Kernel Learning (DKL)** | Wilson et al. (2016) — deep features fed into GP; Tian et al. (2021) — DKL for capacity trajectory with LSTM encoder | High (RMSE ~1–2%) | ✅ Approximate | ❌ | High | Expensive joint training; interpretability fully lost in the deep MLP |
| **BiLSTM / RNN-based** | Zhang et al. (2018) — BiLSTM SOH; Ren et al. (2021) — attention-BiLSTM with sliding window features | High on smooth curves | ❌ Point prediction only | ❌ | Low (overfit to noise) | No uncertainty; fails on out-of-distribution cells; data-hungry |
| **Wide & Deep Neural Networks** | Cheng et al. (2016) — Wide & Deep for memorization + generalization; Bian et al. (2020) — W&D for battery degradation | High on both linear + knee | ❌ | ❌ | Moderate | Black-box; no uncertainty quantification; cannot extrapolate beyond training domain |
| **Bayesian Neural Networks (BNN)** | Gal & Ghahramani (2016) MC-Dropout; Shen et al. (2023) — BNN for SOH with dropout uncertainty | Moderate-High | ✅ Approximate | ❌ | Moderate | Uncertainty poorly calibrated; computationally expensive inference |
| **Physics-Informed NN (PINN)** | Raissi et al. (2019); Shen & Wang (2022) — PINN for SEI layer growth embedded in loss | Moderate | ❌ | Partial (physics ODE) | High | Requires known physics model; fails when underlying mechanism is unknown |
| **Symbolic Regression (EvoSR)** | Schmidt & Lipson (2009) Eureqa; Cranmer (2023) PySR; Chen et al. (2022) — SR for capacity fade law discovery | Moderate (struggles at knees) | ❌ | ✅ Full equation | Low (noisy data inflates tree complexity) | Computationally expensive; degrades catastrophically on noisy or high-uncertainty cycles without weighting |
| **Hybrid GPR + SR (Proposed WD-GPR-SR)** | **This work** — Wide-and-Deep kernel SVGP + uncertainty-weighted symbolic residual correction | **High (target RMSE < 1%)** | **✅ Full calibrated posterior** | **✅ Closed-form residual law** | **High (uncertainty weighting suppresses noisy SR fits)** | Hyperparameter sensitivity; SR search cost $O(P \cdot G \cdot N)$ |

### 1.2 Research Gap Summary

Existing approaches suffer from at least one of three fundamental deficiencies:

1. **Accuracy vs. Interpretability trade-off:** Deep networks achieve high accuracy but are black-boxes. Symbolic regression produces interpretable equations but fails on non-linear degradation knees without proper noise handling.
2. **Missing calibrated uncertainty:** BiLSTM and Wide & Deep networks output point predictions, disqualifying them from safety-critical battery management where uncertainty bounds are mandatory.
3. **Inability to handle heteroscedastic noise:** Standard GPR assumes a single noise hyperparameter $\sigma_n^2$, under-representing cycle-dependent noise that increases near end-of-life.

**WD-GPR-SR addresses all three:** the Wide-and-Deep kernel handles both global linear and local non-linear dynamics, SVGP provides scalable calibrated uncertainty, and the uncertainty-weighted SR loss ensures the discovered equation corrects only the *confident* residual regions.

---

## 2. Complete Mathematical Formulation of WD-GPR-SR

### 2.1 Problem Setup

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ be the training dataset, where:
- $\mathbf{x}_i \in \mathbb{R}^D$ is the feature vector for cycle $i$ (e.g., discharge capacity, internal resistance, temperature, dQ/dV peak features, cycle index)
- $y_i \in [0, 1]$ is the State-of-Health (SOH), normalised as $\text{SOH}_i = C_i / C_{\text{nominal}}$
- $N$ is the total number of cycles across all cells in the training corpus

We model the latent SOH function $f : \mathbb{R}^D \to \mathbb{R}$ as a Gaussian Process:

$$f(\mathbf{x}) \sim \mathcal{GP}\left(\mu_0(\mathbf{x}),\; k_{\text{WD}}(\mathbf{x}, \mathbf{x}')\right)$$

with observations $y_i = f(\mathbf{x}_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$.

---

### 2.2 Wide-and-Deep Covariance Kernel

#### 2.2.1 Definition

The Wide-and-Deep (WD) kernel is defined as a three-component additive kernel:

$$\boxed{k_{\text{WD}}(\mathbf{x}, \mathbf{x}') = \underbrace{\sigma_w^2 \sum_{d=1}^{D} w_d\, x_d x'_d}_{\text{Wide (Linear) Branch}} + \underbrace{\sigma_{\text{ard}}^2 \exp\!\left(-\frac{1}{2}\sum_{d=1}^{D} \frac{(x_d - x'_d)^2}{\ell_d^2}\right)}_{\text{SE-ARD (Smooth Degradation) Branch}} + \underbrace{\sigma_{\text{deep}}^2\; k_{\text{Mat52}}\!\left(\phi_{\boldsymbol{\theta}}(\mathbf{x}),\, \phi_{\boldsymbol{\theta}}(\mathbf{x}')\right)}_{\text{Deep (Non-linear Knee) Branch}}}$$

where the hyperparameter set is $\boldsymbol{\Theta} = \{\sigma_w^2, \{w_d\}_{d=1}^D, \sigma_{\text{ard}}^2, \{\ell_d\}_{d=1}^D, \sigma_{\text{deep}}^2, \boldsymbol{\theta}\}$.

#### 2.2.2 Wide Branch — Global Linear Degradation

The wide branch $k_{\text{wide}}(\mathbf{x}, \mathbf{x}') = \sigma_w^2 \sum_{d=1}^D w_d x_d x'_d$ is a weighted linear dot-product kernel. Its corresponding RKHS consists of all linear functions:

$$f_{\text{wide}}(\mathbf{x}) = \sum_{d=1}^D \beta_d x_d, \quad \beta_d \sim \mathcal{N}(0, \sigma_w^2 w_d)$$

This captures the **monotonic capacity fade** observable in the early and mid-life of lithium-ion cells (typically cycles 1–600 for NASA PCOE B0005–B0018). The per-dimension weights $\{w_d\}$ act as **Automatic Relevance Determination (ARD)** priors: features with large $w_d$ (e.g., discharge capacity, internal resistance) contribute more to the global trend, while irrelevant features (e.g., ambient temperature under controlled experiments) are suppressed.

*Proof of ARD property:* The marginal variance of $f_{\text{wide}}$ in dimension $d$ is $\sigma_w^2 w_d \mathbb{E}[x_d^2]$. As $w_d \to 0$, the posterior mean projection onto dimension $d$ vanishes, effectively removing that feature from the linear model without manual feature selection.

#### 2.2.3 SE-ARD Branch — Smooth Monotone Mid-Life Degradation

The Squared Exponential kernel with ARD length-scales:

$$k_{\text{SE-ARD}}(\mathbf{x}, \mathbf{x}') = \sigma_{\text{ard}}^2 \exp\!\left(-\frac{1}{2} \sum_{d=1}^D \frac{(x_d - x'_d)^2}{\ell_d^2}\right)$$

models **smooth, slowly-varying** correlations in the degradation trajectory. The length-scale $\ell_d$ controls the characteristic correlation length in dimension $d$:
- Large $\ell_d$: slow variation → captures long-range capacity drift
- Small $\ell_d$: fast variation → captures local cycle-to-cycle fluctuations

The ARD length-scales are learned via Type-II MLE (marginal likelihood optimisation), providing principled automatic feature selection without cross-validation.

#### 2.2.4 Deep Branch — Non-Linear Knee-Point Dynamics

The deep branch projects raw features through a trained MLP $\phi_{\boldsymbol{\theta}} : \mathbb{R}^D \to \mathbb{R}^H$ before computing a Matérn-5/2 kernel in the learned feature space:

$$k_{\text{Mat52}}(\mathbf{u}, \mathbf{v}) = \sigma_{\text{deep}}^2 \left(1 + \frac{\sqrt{5}\,\|\mathbf{u} - \mathbf{v}\|_2}{\rho} + \frac{5\|\mathbf{u} - \mathbf{v}\|_2^2}{3\rho^2}\right) \exp\!\left(-\frac{\sqrt{5}\,\|\mathbf{u} - \mathbf{v}\|_2}{\rho}\right)$$

where $\mathbf{u} = \phi_{\boldsymbol{\theta}}(\mathbf{x})$, $\mathbf{v} = \phi_{\boldsymbol{\theta}}(\mathbf{x}')$, and $\rho$ is the Matérn length-scale in feature space.

The Matérn-5/2 kernel is chosen over SE because it produces **twice mean-square differentiable** sample paths — sufficient smoothness to model capacity regeneration humps but with heavier tails than SE, enabling sharper knee-point transitions around SOH $\approx$ 0.80.

The MLP architecture $\phi_{\boldsymbol{\theta}}$ with layers $[D \to 64 \to 64 \to H]$ and ReLU activations learns a **non-linear warp** of the input space that disentangles the knee-point regime from the linear degradation regime, a transformation impossible with any fixed analytic kernel.

#### 2.2.5 Positive Semi-Definiteness of $k_{\text{WD}}$

**Claim:** $k_{\text{WD}}$ is a valid kernel (positive semi-definite).

**Proof:** A sum of valid kernels is a valid kernel (closure of PSD functions under addition). We verify each branch:
- $k_{\text{wide}}$: the linear kernel $k(\mathbf{x},\mathbf{x}') = \mathbf{x}^\top \mathbf{W} \mathbf{x}'$ with $\mathbf{W} = \sigma_w^2 \text{diag}(w_1,\ldots,w_D) \succ 0$ is PSD. ✓
- $k_{\text{SE-ARD}}$: the squared exponential kernel is PSD by Bochner's theorem. ✓
- $k_{\text{deep}}$: composition of a PSD kernel with a deterministic feature map $\phi_{\boldsymbol{\theta}}$ is PSD by the kernel substitution lemma. ✓

Therefore $k_{\text{WD}} = k_{\text{wide}} + k_{\text{SE-ARD}} + k_{\text{deep}}$ is PSD. $\blacksquare$

---

### 2.3 Sparse Variational Gaussian Process (SVGP) Objective

#### 2.3.1 Motivation: Scalability

Exact GPR posterior inference requires solving a linear system with the $N \times N$ kernel matrix $\mathbf{K}_{NN}$, incurring $O(N^3)$ time and $O(N^2)$ memory. For $N = 10{,}000$ cycles across a multi-cell NASA/CALCE corpus, this is computationally prohibitive. Sparse variational inference with $M \ll N$ inducing points reduces complexity to $O(NM^2 + M^3)$.

#### 2.3.2 Inducing Point Setup

Introduce $M$ inducing inputs $\mathbf{Z} = \{\mathbf{z}_m\}_{m=1}^M \subset \mathbb{R}^D$ and the corresponding inducing function values $\mathbf{u} = f(\mathbf{Z}) \in \mathbb{R}^M$.

Define the following kernel matrices:
- $\mathbf{K}_{MM} = k_{\text{WD}}(\mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{M \times M}$ — inducing-inducing kernel
- $\mathbf{K}_{NM} = k_{\text{WD}}(\mathbf{X}, \mathbf{Z}) \in \mathbb{R}^{N \times M}$ — data-inducing kernel  
- $\mathbf{K}_{NN} = k_{\text{WD}}(\mathbf{X}, \mathbf{X}) \in \mathbb{R}^{N \times N}$ — full data kernel (never explicitly formed)

#### 2.3.3 Variational Distribution

The true posterior $p(\mathbf{u} \mid \mathbf{y})$ is intractable when combined with the deep branch. We introduce a variational distribution over inducing values:

$$q(\mathbf{u}) = \mathcal{N}(\mathbf{u};\; \mathbf{m},\; \mathbf{S}), \quad \mathbf{m} \in \mathbb{R}^M,\; \mathbf{S} \in \mathbb{R}^{M \times M},\; \mathbf{S} \succ 0$$

The joint variational distribution over latent function values at training inputs $\mathbf{f} = f(\mathbf{X})$ is obtained by marginalising out $\mathbf{u}$:

$$q(\mathbf{f}) = \int p(\mathbf{f} \mid \mathbf{u})\, q(\mathbf{u})\, d\mathbf{u} = \mathcal{N}(\mathbf{f};\; \boldsymbol{\mu}_q,\; \boldsymbol{\Sigma}_q)$$

where, using the GP conditional $p(\mathbf{f} \mid \mathbf{u}) = \mathcal{N}(\mathbf{K}_{NM}\mathbf{K}_{MM}^{-1}\mathbf{u},\; \mathbf{K}_{NN} - \mathbf{Q}_{NN})$ with $\mathbf{Q}_{NN} = \mathbf{K}_{NM}\mathbf{K}_{MM}^{-1}\mathbf{K}_{MN}$:

$$\boldsymbol{\mu}_q = \mathbf{K}_{NM}\mathbf{K}_{MM}^{-1}\mathbf{m}$$

$$\boldsymbol{\Sigma}_q = \mathbf{K}_{NN} - \mathbf{Q}_{NN} + \mathbf{K}_{NM}\mathbf{K}_{MM}^{-1}\mathbf{S}\mathbf{K}_{MM}^{-1}\mathbf{K}_{MN}$$

#### 2.3.4 Evidence Lower Bound (ELBO)

Maximising the marginal likelihood $\log p(\mathbf{y})$ is intractable. Instead, we maximise the **ELBO** $\mathcal{L}_{\text{ELBO}}$, a lower bound obtained via Jensen's inequality:

$$\log p(\mathbf{y}) = \log \int p(\mathbf{y} \mid \mathbf{f})\,p(\mathbf{f})\,d\mathbf{f} \geq \mathcal{L}_{\text{ELBO}}$$

Using the standard variational decomposition:

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q(\mathbf{f})}\!\left[\log p(\mathbf{y} \mid \mathbf{f})\right]}_{\text{Expected log-likelihood}} - \underbrace{D_{\text{KL}}\!\left[q(\mathbf{u}) \;\|\; p(\mathbf{u})\right]}_{\text{KL regulariser}}$$

For Gaussian likelihood $p(y_i \mid f_i) = \mathcal{N}(y_i;\, f_i,\, \sigma_n^2)$, the expected log-likelihood decomposes as:

$$\mathbb{E}_{q(\mathbf{f})}\!\left[\log p(\mathbf{y} \mid \mathbf{f})\right] = \sum_{i=1}^N \log \mathcal{N}\!\left(y_i;\, [\boldsymbol{\mu}_q]_i,\, \sigma_n^2\right) - \frac{1}{2\sigma_n^2} \sum_{i=1}^N [\boldsymbol{\Sigma}_q]_{ii}$$

$$= -\frac{N}{2}\log(2\pi\sigma_n^2) - \frac{1}{2\sigma_n^2}\left(\|\mathbf{y} - \boldsymbol{\mu}_q\|^2 + \text{tr}(\boldsymbol{\Sigma}_q)\right)$$

The KL divergence between two Gaussians is:

$$D_{\text{KL}}[q(\mathbf{u}) \| p(\mathbf{u})] = \frac{1}{2}\left[\text{tr}(\mathbf{K}_{MM}^{-1}\mathbf{S}) + \mathbf{m}^\top \mathbf{K}_{MM}^{-1}\mathbf{m} - M + \log\frac{|\mathbf{K}_{MM}|}{|\mathbf{S}|}\right]$$

Putting it together, the **full SVGP ELBO** is:

$$\boxed{\mathcal{L}_{\text{ELBO}} = -\frac{N}{2}\log(2\pi\sigma_n^2) - \frac{\|\mathbf{y} - \mathbf{K}_{NM}\mathbf{K}_{MM}^{-1}\mathbf{m}\|^2 + \text{tr}(\boldsymbol{\Sigma}_q)}{2\sigma_n^2} - \frac{1}{2}\left[\text{tr}(\mathbf{K}_{MM}^{-1}\mathbf{S}) + \mathbf{m}^\top\mathbf{K}_{MM}^{-1}\mathbf{m} - M + \log\frac{|\mathbf{K}_{MM}|}{|\mathbf{S}|}\right]}$$

#### 2.3.5 Complexity Analysis

| Operation | Exact GPR | SVGP |
|---|---|---|
| Kernel matrix formation | $O(N^2 D)$ | $O(NM D)$ |
| Cholesky factorisation | $O(N^3)$ | $O(M^3)$ |
| Posterior mean prediction | $O(N^2)$ | $O(NM)$ |
| Memory | $O(N^2)$ | $O(NM)$ |
| **Total training** | **$O(N^3)$** | **$O(NM^2 + M^3)$** |

For $N = 10{,}000$ and $M = 200$: exact GPR requires $\sim 10^{12}$ FLOPs; SVGP requires $\sim 4 \times 10^8$ FLOPs — a **2500× reduction**.

#### 2.3.6 Mini-Batch ELBO for Stochastic Optimisation

Since each data term $\log p(y_i \mid f_i)$ is independent given $\mathbf{u}$, the expected log-likelihood can be unbiasedly estimated on a mini-batch $\mathcal{B} \subset [N]$, $|\mathcal{B}| = B$:

$$\mathcal{L}_{\text{ELBO}}^{\mathcal{B}} = \frac{N}{B}\sum_{i \in \mathcal{B}} \mathbb{E}_{q(f_i)}\!\left[\log p(y_i \mid f_i)\right] - D_{\text{KL}}[q(\mathbf{u}) \| p(\mathbf{u})]$$

This enables **stochastic gradient descent** through mini-batches, further reducing per-step cost to $O(BM^2)$ with $B \ll N$.

---

### 2.4 Uncertainty-Weighted Symbolic Residual Correction

#### 2.4.1 GPR Posterior and Residuals

After optimising $\mathcal{L}_{\text{ELBO}}$ with respect to $(\mathbf{m}, \mathbf{S}, \boldsymbol{\Theta})$, the WD-SVGP posterior predictive distribution at a test input $\mathbf{x}_*$ is:

$$p(f_* \mid \mathbf{x}_*, \mathcal{D}) = \mathcal{N}\!\left(\mu_{\text{WD}}(\mathbf{x}_*),\; \sigma_{\text{WD}}^2(\mathbf{x}_*)\right)$$

with posterior mean and variance:

$$\mu_{\text{WD}}(\mathbf{x}_*) = \mathbf{k}_{*M}\mathbf{K}_{MM}^{-1}\mathbf{m}$$

$$\sigma_{\text{WD}}^2(\mathbf{x}_*) = k_{\text{WD}}(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_{*M}\left(\mathbf{K}_{MM}^{-1} - \mathbf{K}_{MM}^{-1}\mathbf{S}\mathbf{K}_{MM}^{-1}\right)\mathbf{k}_{M*}$$

where $\mathbf{k}_{*M} = k_{\text{WD}}(\mathbf{x}_*, \mathbf{Z}) \in \mathbb{R}^{1 \times M}$.

The **structural residuals** at each training point are:

$$r_i = y_i - \mu_{\text{WD}}(\mathbf{x}_i), \quad i = 1, \ldots, N$$

These residuals encode the systematic structure that the WD-GPR failed to explain — principally the **capacity regeneration humps** and **abrupt knee-point transitions** that are difficult to express purely in kernel form.

#### 2.4.2 Uncertainty-Weighted SR Loss Function

We search for a symbolic function $g : \mathbb{R}^D \to \mathbb{R}$ (a tree of mathematical primitives: $\{+, -, \times, \div, \exp, \log, \sin, x^n\}$) that fits these residuals. The evolutionary search optimises:

$$\boxed{\mathcal{L}_{\text{SR}}(g) = \frac{1}{N}\sum_{i=1}^{N} \frac{\left(r_i - g(\mathbf{x}_i)\right)^2}{\sigma_{\text{WD}}^2(\mathbf{x}_i) + \epsilon} + \lambda \cdot \text{Nodes}(g)}$$

where:
- $\sigma_{\text{WD}}^2(\mathbf{x}_i) + \epsilon$ is the posterior variance regularised by $\epsilon > 0$ (typically $\epsilon = 10^{-6}$) for numerical stability
- $\text{Nodes}(g)$ counts the total nodes in the expression tree of $g$, penalising complexity
- $\lambda > 0$ controls the complexity-accuracy trade-off (Pareto front exploration)

#### 2.4.3 MLE Justification of the Uncertainty Weighting

**Theorem:** Minimising $\mathcal{L}_{\text{SR}}(g)$ (without the complexity term) is equivalent to Maximum Likelihood Estimation under a heteroscedastic Gaussian noise model where the noise variance at cycle $i$ equals $\sigma_{\text{WD}}^2(\mathbf{x}_i) + \epsilon$.

**Proof:**

*Step 1 — Heteroscedastic noise model.* Assume the residuals follow:

$$r_i \mid g(\mathbf{x}_i) \sim \mathcal{N}\!\left(g(\mathbf{x}_i),\; \tilde{\sigma}_i^2\right), \quad \tilde{\sigma}_i^2 = \sigma_{\text{WD}}^2(\mathbf{x}_i) + \epsilon$$

*Step 2 — Log-likelihood.* The log-likelihood of the residual dataset $\{r_i\}_{i=1}^N$ given $g$ is:

$$\log \mathcal{L}(g) = \sum_{i=1}^N \log p(r_i \mid g(\mathbf{x}_i),\; \tilde{\sigma}_i^2)$$

$$= \sum_{i=1}^N \left[-\frac{1}{2}\log(2\pi\tilde{\sigma}_i^2) - \frac{(r_i - g(\mathbf{x}_i))^2}{2\tilde{\sigma}_i^2}\right]$$

$$= -\frac{N}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^N \log(\tilde{\sigma}_i^2) - \frac{1}{2}\sum_{i=1}^N \frac{(r_i - g(\mathbf{x}_i))^2}{\tilde{\sigma}_i^2}$$

*Step 3 — MLE reduction.* Since $\tilde{\sigma}_i^2$ is fixed (determined by the trained WD-SVGP, not a free parameter of the SR stage), the terms $-\frac{N}{2}\log(2\pi)$ and $-\frac{1}{2}\sum_i \log(\tilde{\sigma}_i^2)$ are **constants** with respect to $g$. Therefore:

$$\hat{g}_{\text{MLE}} = \arg\max_g \log \mathcal{L}(g) = \arg\min_g \sum_{i=1}^N \frac{(r_i - g(\mathbf{x}_i))^2}{\sigma_{\text{WD}}^2(\mathbf{x}_i) + \epsilon}$$

$$= \arg\min_g \frac{1}{N}\sum_{i=1}^N \frac{(r_i - g(\mathbf{x}_i))^2}{\sigma_{\text{WD}}^2(\mathbf{x}_i) + \epsilon} = \arg\min_g \mathcal{L}_{\text{SR}}(g)\bigg|_{\lambda=0}$$

This proves that minimising $\mathcal{L}_{\text{SR}}$ (without complexity penalty) is exactly MLE under heteroscedastic noise. $\blacksquare$

#### 2.4.4 Why Uncertainty Weighting Prevents Symbolic Regression Overfitting

**Corollary:** The weighting by $(\sigma_{\text{WD}}^2(\mathbf{x}_i) + \epsilon)^{-1}$ suppresses SR overfitting to noisy, high-uncertainty cycles.

**Argument:**

Consider cycle $i^*$ near the knee-point (SOH $\approx$ 0.80), where rapid degradation induces high measurement noise and model uncertainty:

$$\sigma_{\text{WD}}^2(\mathbf{x}_{i^*}) \gg \sigma_{\text{WD}}^2(\mathbf{x}_j) \quad \text{for typical mid-life cycle } j$$

In an **unweighted** SR loss, the term $(r_{i^*} - g(\mathbf{x}_{i^*}))^2$ contributes equally to all cycles. If $r_{i^*}$ is large due to stochastic noise rather than systematic structure, SR will grow complex tree branches solely to reduce this one term — a form of **overfitting to noise**.

In the **uncertainty-weighted** loss, the contribution of cycle $i^*$ is:

$$\frac{(r_{i^*} - g(\mathbf{x}_{i^*}))^2}{\sigma_{\text{WD}}^2(\mathbf{x}_{i^*}) + \epsilon} \ll \frac{(r_{i^*} - g(\mathbf{x}_{i^*}))^2}{\epsilon}$$

The effective weight of cycle $i^*$ in the total loss is **attenuated** proportionally to its GPR-estimated variance. SR therefore allocates its representational budget to cycles where:
1. The residual $|r_i|$ is large (systematic error), **and**
2. The GPR uncertainty $\sigma_{\text{WD}}^2(\mathbf{x}_i)$ is small (trustworthy estimate)

This is precisely the **information-theoretically optimal** allocation: fit structure where you are confident, and ignore noise where you are uncertain. The complexity penalty $\lambda \cdot \text{Nodes}(g)$ further enforces Occam's Razor, preventing the SR expression tree from bloating to accommodate individual noisy outlier cycles.

#### 2.4.5 Final WD-GPR-SR Prediction

The final SOH prediction combines the GPR posterior mean with the symbolic correction:

$$\hat{y}(\mathbf{x}_*) = \mu_{\text{WD}}(\mathbf{x}_*) + \hat{g}(\mathbf{x}_*)$$

with the **total predictive uncertainty** inherited from the GPR posterior:

$$\text{Var}\!\left[\hat{y}(\mathbf{x}_*)\right] = \sigma_{\text{WD}}^2(\mathbf{x}_*) + \sigma_n^2$$

The symbolic function $\hat{g}$ is deterministic (post-SR-search), so it does not inflate the uncertainty — it only corrects the mean. The closed-form expression of $\hat{g}$ provides **physical interpretability**: the discovered equation reveals which feature combinations drive systematic residuals, potentially identifying new degradation mechanisms not captured by the kernel prior.

---

## 3. Summary of Mathematical Notation

| Symbol | Definition |
|---|---|
| $N$ | Number of training cycles |
| $M$ | Number of inducing points ($M \ll N$) |
| $D$ | Feature dimensionality |
| $\mathbf{x}_i \in \mathbb{R}^D$ | Feature vector for cycle $i$ |
| $y_i \in [0,1]$ | Observed SOH at cycle $i$ |
| $k_{\text{WD}}(\cdot, \cdot)$ | Wide-and-Deep covariance kernel |
| $\phi_{\boldsymbol{\theta}}(\cdot)$ | Deep MLP feature extractor (parameters $\boldsymbol{\theta}$) |
| $\mathbf{Z} \in \mathbb{R}^{M \times D}$ | Inducing input locations |
| $\mathbf{m} \in \mathbb{R}^M$ | Variational mean of inducing outputs |
| $\mathbf{S} \in \mathbb{R}^{M \times M}$ | Variational covariance of inducing outputs |
| $\mu_{\text{WD}}(\mathbf{x})$ | WD-SVGP posterior mean |
| $\sigma_{\text{WD}}^2(\mathbf{x})$ | WD-SVGP posterior variance |
| $r_i$ | Structural residual at cycle $i$ |
| $g(\cdot)$ | Symbolic regression correction function |
| $\mathcal{L}_{\text{SR}}(g)$ | Uncertainty-weighted SR loss |
| $\lambda$ | SR complexity penalty coefficient |
| $\text{Nodes}(g)$ | Number of nodes in expression tree of $g$ |
| $\epsilon$ | Numerical stability constant ($10^{-6}$) |

