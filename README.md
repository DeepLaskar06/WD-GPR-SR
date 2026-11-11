# WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge AI](https://img.shields.io/badge/NVIDIA%20Jetson-Orin%20Nano%20Certified-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

**Author:** Moonis Laskar  
**Roll No:** 2401CS29  
**Institution:** Indian Institute of Technology Patna (IIT Patna), Department of Computer Science & Engineering  
**Project:** B.Tech Project-Based Learning (PBL) in Machine Learning  

---

## 📌 Executive Summary

Lithium-ion battery degradation is non-linear, multi-physical, and safety-critical. Deep neural networks achieve low training errors but produce uncalibrated predictions that fail silently under unseen distributions, risking catastrophic thermal runaway. Conversely, standard Gaussian Process Regression (GPR) provides rigorous Bayesian uncertainty but suffers from an $\mathcal{O}(N^3)$ computational bottleneck ($\sim 1,420\text{ ms}$ per inference), severely violating automotive Battery Management System (BMS) hard real-time deadlines ($\le 100\text{ ms}$).

**WD-GPR-SR** solves this dilemma through a tri-hybrid architecture:
1. **Wide Linear Branch:** Learns Automatic Relevance Determination (ARD) covariance weights over health indicators (HIs) to model global monotonic degradation.
2. **Deep Manifold Branch:** Feeds features through a non-linear MLP ($\phi(\mathbf{x}): 10 \to 16 \to 8$, SiLU) coupled with a Matérn-5/2 kernel to capture electro-thermal dynamics and capacity regeneration jumps.
3. **Sparse Inducing Points ($M=64$):** Slashes variational complexity from $\mathcal{O}(N^3)$ down to $\mathcal{O}(M^2 N)$, delivering sub-millisecond Bayesian inference ($0.110\text{ ms}$).
4. **Uncertainty-Weighted Symbolic Regression (UW-SR):** Uses the GPR predictive epistemic variance as inverse-variance weights:
   $$w_i = \frac{1}{\sigma_{\mathrm{GP}}^2(\mathbf{x}_i) + \epsilon}$$
   guiding genetic Pareto search to extract closed-form physics equations from the residual while strictly bounding corrections to $\pm 3\sigma_{\mathrm{GP}}$.

---

## 🚀 Key Quantitative Results (Strict LOCO Benchmark)

All models evaluated under strict **Leave-One-Cell-Out (LOCO)** cross-validation (zero temporal or inter-cell data leakage) over 3 deterministic seeds (`[42, 100, 2026]`):

| Dataset | Model | RMSE (%) | MAE (%) | $R^2$ | MAPE (%) | PICP (%) | NLL |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NASA** | SVR (RBF) | 2.871 | 2.098 | 0.8924 | 2.961 | -- | -- |
| **NASA** | Random Forest | 1.337 | 1.011 | 0.9780 | 1.377 | -- | -- |
| **NASA** | XGBoost | 1.300 | 0.997 | 0.9789 | 1.356 | -- | -- |
| **NASA** | Exact GPR (Matérn) | 21.598 | 18.883 | -5.151 | 25.064 | 14.6% | 91.00 |
| **NASA** | Deep-Kernel GPR | 3.142 | 2.693 | 0.8535 | 3.592 | 56.5% | 8.94 |
| **NASA** | **WD-GPR-SR (Proposed)** | **0.992** | **0.878** | **0.9876** | **1.157** | **98.8%** | **1.48** |
| **CALCE** | SVR (RBF) | 4.092 | 2.923 | 0.8872 | 4.908 | -- | -- |
| **CALCE** | Random Forest | 1.968 | 0.965 | 0.9719 | 1.761 | -- | -- |
| **CALCE** | Exact GPR (Matérn) | 13.784 | 6.471 | -0.046 | 11.666 | 67.2% | 49.33 |
| **CALCE** | Deep-Kernel GPR | 1.867 | 0.900 | 0.9784 | 1.663 | 91.3% | 4.23 |
| **CALCE** | **WD-GPR-SR (Proposed)** | **0.130** | **0.058** | **0.9999** | **0.104** | **100.0%** | **0.99** |

---

## ⚡ Edge AI Deployment on NVIDIA Jetson Orin Nano

Benchmarked on **NVIDIA Jetson Orin Nano Developer Kit (MAXN 15W TDP)**:
- **Inference Latency:** **0.110 ms** (Standard GPR: $1,420\text{ ms}$ — **12,900× Speedup**, well below BMS $100\text{ ms}$ deadline).
- **Energy per Inference:** **0.00148 J (1.48 mJ)** (Standard GPR: $13.135\text{ J}$ — **8,880× Energy Reduction**).
- **Throughput:** **6,237 Hz** (real-time stream processing).
- **RAM Footprint:** **148.5 MB** peak memory usage.

---

## 📂 Repository Structure

```text
├── 2401CS29_MLProject.ipynb          # Full, self-contained, executed Jupyter Notebook (Phases 1-7)
├── 2401CS29_MLProject.pdf            # Comprehensive Academic Project Report (ReportLab)
├── 2401CS29_MLProject.pptx           # Professional Conference Slide Deck (python-pptx)
├── 2401CS29_Project_Poster.pdf       # 48"x36" Academic Research Poster (300 DPI, TrueType Embedded)
├── build_poster.bat / .sh            # One-click poster compilation scripts
├── requirements.txt                  # Python dependencies
├── data/
│   └── processed/                    # Cleaned feature tables (features_all.csv, ic_features.csv, etc.)
├── docs/
│   ├── literature_review_and_math.md # Theoretical proofs and kernel formulations
│   └── poster_template/              # HTML/CSS & LaTeX beamerposter templates
├── jetson/
│   ├── benchmark_orin_nano.py        # Edge benchmarking script
│   ├── export_model.py               # ONNX & TensorRT export pipeline
│   └── symbolic_residual.c           # High-speed static C implementation of discovered equations
├── models/                           # Exported ONNX, TensorRT engines, alpha & inducing vectors
├── results/
│   ├── figures/                      # High-res publication figures (.png)
│   ├── poster_assets/                # 300 DPI sub-plots & TrueType math panels
│   ├── tables/                       # Quantitative CSV tables
│   └── discovered_equations.txt      # Pareto-optimal symbolic equations discovered
└── src/                              # Python modules (data loaders, training, XAI, compilation)
```

---

## 🔬 Reproducibility & Quickstart

### 1. Installation
```bash
git clone https://github.com/DeepLaskar06/WD-GPR-SR.git
cd WD-GPR-SR
pip install -r requirements.txt
```

### 2. Run Poster Generation
To regenerate the 48" $\times$ 36" academic conference poster with TrueType font embedding and mathematical equation panels:
```bash
# Windows
.\build_poster.bat

# Linux / macOS
./build_poster.sh
```

### 3. Run Edge Benchmarks
```bash
python jetson/benchmark_orin_nano.py
```

---

## 📜 Citation & Academic Contact

**Moonis Laskar**  
Department of Computer Science & Engineering  
Indian Institute of Technology Patna (IIT Patna), Bihta, Bihar 801106, India  
Email: `moonis_2401cs29@iitp.ac.in`  
GitHub: [@DeepLaskar06](https://github.com/DeepLaskar06)
