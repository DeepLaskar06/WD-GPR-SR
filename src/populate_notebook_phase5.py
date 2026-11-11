"""
Populate Jupyter Notebook with Phase 5 Results (Explainable AI & Ablation Study).
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction
"""

import json
import base64
from pathlib import Path
import pandas as pd

nb_path = Path("2401CS29_MLProject.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

fig_dir = Path("results/figures")

def get_b64(fname):
    p = fig_dir / fname
    if p.exists():
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

b64_shap = get_b64("shap_summary.png")
b64_lime = get_b64("lime_explanation.png")
b64_pdp = get_b64("pdp_ice.png")
b64_ard = get_b64("ard_relevance.png")
b64_sens = get_b64("symbolic_sensitivities.png")
b64_abl = get_b64("ablation_study_comparison.png")

# Load ablation study table
abl_csv = Path("results/tables/ablation_study.csv")
df_abl = pd.read_csv(abl_csv)

# -------------------------------------------------------------
# Construct Section 8: Explainable AI (XAI) Suite
# -------------------------------------------------------------
sec8_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Section 8: Explainable AI (XAI) Comprehensive Interpretability Analysis (Days 21–22) <a id=\"section-8\"></a>\n\n",
        "**Author:** Moonis Laskar (Roll No: `2401CS29`)  \n",
        "**Project:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  \n\n",
        "### 8.1 Multi-Scale Interpretability Framework\n",
        "Modern deep learning architectures for battery prognostics frequently suffer from the *black-box dilemma*: deep models deliver high empirical accuracy but offer zero insight into degradation physics, preventing certification in aerospace and automotive Battery Management Systems (BMS). Conversely, purely empirical equivalent circuit models lack the capacity to capture non-linear degradation knee points.\n\n",
        "To establish complete scientific transparency, we subject the proposed WD-GPR-SR model to **5 complementary Explainable AI (XAI) modalities**, centered on the NASA test cell `B0018`:\n\n",
        "1. **SHAP Beeswarm Summary Plot (`shap.KernelExplainer`):** Quantifies the global marginal contribution and directional impact of every health indicator on predicted battery SOH.\n",
        "2. **LIME Local Instance Explanation (`lime.lime_tabular`):** Builds a locally faithful linear surrogate explaining why an end-of-life battery cycle (SOH < 75%) was flagged as heavily degraded.\n",
        "3. **Partial Dependence (PDP) & Individual Conditional Expectation (ICE):** Visualizes the non-linear marginal response curves and sample-level variation for the top 2 Health Indicators (`cc_charge_time` and `ic_peak_height`).\n",
        "4. **Global ARD Length-Scale Relevance ($1/\\ell_j$ / Variance $v_j$):** Extracts the Bayesian feature importance weights directly from the Wide Linear branch and ARD kernel prior.\n",
        "5. **Dynamic Lifecycle Sensitivity Trajectories ($|\\partial \\hat{y} / \\partial x_j|$):** Computes the evolution of analytical gradient sensitivities across all 132 operating cycles, capturing the abrupt sensitivity surge at the onset of the **degradation knee point** (Cycle ~100, SOH < 80%).\n"
    ]
}

sec8_code_source = [
    "# =====================================================================\n",
    "# Section 8 Execution: Explainable AI (XAI) Visualizations & Insights\n",
    "# =====================================================================\n",
    "from IPython.display import Image, display\n",
    "from pathlib import Path\n\n",
    "fig_dir = Path(\"results/figures\")\n\n",
    "xai_figures = [\n",
    "    (\"shap_summary.png\", \"1. Global Feature Attribution & Directionality (SHAP Beeswarm Summary)\", 1000),\n",
    "    (\"lime_explanation.png\", \"2. Local Feature Attributions for Severely Degraded Cycle SOH < 75% (LIME)\", 980),\n",
    "    (\"pdp_ice.png\", \"3. Partial Dependence (PDP) & Individual Conditional Expectation (ICE) Curves\", 1050),\n",
    "    (\"ard_relevance.png\", \"4. Bayesian ARD Relevance Ranking across Health Indicators\", 980),\n",
    "    (\"symbolic_sensitivities.png\", \"5. Dynamic Lifecycle Sensitivity Trajectories Across Knee-Point Transition\", 1100),\n",
    "]\n\n",
    "for fname, title, w in xai_figures:\n",
    "    p = fig_dir / fname\n",
    "    if p.exists():\n",
    "        print(\"=\" * 100)\n",
    "        print(f\" [XAI] {title} ({fname})\")\n",
    "        print(\"=\" * 100)\n",
    "        display(Image(filename=str(p), width=w))\n"
]

sec8_outputs = []
xai_items = [
    ("shap_summary.png", "1. Global Feature Attribution & Directionality (SHAP Beeswarm Summary)", 1000),
    ("lime_explanation.png", "2. Local Feature Attributions for Severely Degraded Cycle SOH < 75% (LIME)", 980),
    ("pdp_ice.png", "3. Partial Dependence (PDP) & Individual Conditional Expectation (ICE) Curves", 1050),
    ("ard_relevance.png", "4. Bayesian ARD Relevance Ranking across Health Indicators", 980),
    ("symbolic_sensitivities.png", "5. Dynamic Lifecycle Sensitivity Trajectories Across Knee-Point Transition", 1100),
]

for fname, title, w in xai_items:
    b64_data = get_b64(fname)
    if b64_data:
        sec8_outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": ["=" * 100 + "\n", f" [XAI] {title} ({fname})\n", "=" * 100 + "\n"]
        })
        sec8_outputs.append({
            "data": {
                "image/png": b64_data,
                "text/plain": ["<IPython.core.display.Image object>"]
            },
            "metadata": {"image/png": {"width": w}},
            "output_type": "display_data"
        })

sec8_code = {
    "cell_type": "code",
    "execution_count": 14,
    "metadata": {},
    "outputs": sec8_outputs,
    "source": sec8_code_source
}

# -------------------------------------------------------------
# Construct Section 9: Ablation Study
# -------------------------------------------------------------
sec9_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Section 9: Ablation Study & Empirical Architectural Validation (Days 19–20) <a id=\"section-9\"></a>\n\n",
        "**Author:** Moonis Laskar (Roll No: `2401CS29`)  \n",
        "**Project:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  \n\n",
        "### 9.1 Systematic Architectural Decomposition\n",
        "To rigorously isolate and quantify the specific value added by each mathematical component of WD-GPR-SR, we conducted a 4-variant ablation study across all 8 Leave-One-Cell-Out (LOCO) folds and 3 seeds (`[42, 100, 2026]`, 96 runs total):\n\n",
        "1. **Variant A (Deep-Only GPR):** Standard Deep Kernel Learning (MLP $\\mathbb{R}^{10} \\to \\mathbb{R}^8$ + Matérn-5/2 covariance) without the Wide Linear branch and without Symbolic Regression.\n",
        "2. **Variant B (WD-GPR without SR):** The proposed Wide-and-Deep Sparse GPR ($k_{\\text{wide}} + k_{\\text{deep}} + k_{\\text{SE}}$), skipping the residual Symbolic Regression stage.\n",
        "3. **Variant C (WD-GPR + Unweighted SR):** The full architecture, but training the Symbolic Regression engine with standard unweighted MSE (ignoring predictive variance $\\sigma_i^2$).\n",
        "4. **Variant D (Full WD-GPR-SR):** The complete proposed architecture combining the Wide-and-Deep prior with heteroscedastic uncertainty-weighted symbolic residual correction ($w_i = 1 / (\\sigma_i^2 + 10^{-6})$).\n\n",
        "### 9.2 Key Ablation Findings\n",
        "- **Impact of the Wide Linear Prior (Variant A → Variant B):** On NASA Ames, omitting the Wide branch causes Deep-Only GPR to suffer from cross-cell distribution shifts (RMSE rises to **$2.819\\%$**, PICP drops to **$71.0\\%$**). Introducing the Wide Linear kernel forces macroscopic monotonic degradation tracking, reducing test RMSE to **$0.989\\%$** (a **$64.9\\%$ error reduction**) and restoring calibrated 95% coverage to **$98.8\\%$**! On CALCE, Deep-Only GPR degrades to **$7.023\\%$** RMSE (spiking to $16.6\\%$ on `CS2_36`), while the Wide-and-Deep model resolves this completely, achieving an RMSE of **$0.091\\%$** (over **$77\\times$ error reduction**).\n",
        "- **Impact of Heteroscedastic Weighting (Variant C → Variant D):** Weighting the symbolic residual regression by $1/\\sigma_i^2$ prevents genetic expression trees from overfitting anomalous capacity regeneration spikes and transient sensor noise, ensuring the discovered equations reflect true electrophysical mechanics.\n"
    ]
}

sec9_code_source = [
    "# =====================================================================\n",
    "# Section 9 Execution: Ablation Study Comparison (Table & Publication Figure)\n",
    "# =====================================================================\n",
    "import pandas as pd\n",
    "from IPython.display import Image, display\n",
    "from pathlib import Path\n\n",
    "tab_dir = Path(\"results/tables\")\n",
    "fig_dir = Path(\"results/figures\")\n\n",
    "# 1. Load and Display Aggregated Ablation Study Table\n",
    "abl_csv = tab_dir / \"ablation_study.csv\"\n",
    "if abl_csv.exists():\n",
    "    df_abl = pd.read_csv(abl_csv)\n",
    "    for ds_name in [\"NASA\", \"CALCE\"]:\n",
    "        sub = df_abl[df_abl[\"Dataset\"] == ds_name][[\"Variant\", \"RMSE\", \"MAE\", \"R2\", \"NLL\", \"PICP_95\", \"MPIW_95\"]]\n",
    "        print(\"=\" * 115)\n",
    "        print(f\" {ds_name} Ablation Study: Progressive Architectural Impact (Mean ± Std Across LOCO Folds & 3 Seeds)\")\n",
    "        print(\"=\" * 115)\n",
    "        display(sub)\n\n",
    "# 2. Render Publication Ablation Bar Chart\n",
    "fig_abl = fig_dir / \"ablation_study_comparison.png\"\n",
    "if fig_abl.exists():\n",
    "    print(f\"\\n[OK] Rendering Ablation Study Comparison Bar Chart ({fig_abl.name}):\")\n",
    "    display(Image(filename=str(fig_abl), width=1050))\n"
]

sec9_outputs = []
disp_cols = ["Variant", "RMSE", "MAE", "R2", "NLL", "PICP_95", "MPIW_95"]

for ds_name in ["NASA", "CALCE"]:
    sub_abl = df_abl[df_abl["Dataset"] == ds_name][disp_cols]
    sec9_outputs.append({
        "name": "stdout",
        "output_type": "stream",
        "text": [
            "=" * 115 + "\n",
            f" {ds_name} Ablation Study: Progressive Architectural Impact (Mean ± Std Across LOCO Folds & 3 Seeds)\n",
            "=" * 115 + "\n"
        ]
    })
    sec9_outputs.append({
        "data": {
            "text/html": [sub_abl.to_html(index=False, classes="table table-striped table-bordered")],
            "text/plain": [sub_abl.to_string(index=False)]
        },
        "metadata": {},
        "output_type": "display_data"
    })

if b64_abl:
    sec9_outputs.append({
        "name": "stdout",
        "output_type": "stream",
        "text": ["\n[OK] Rendering Ablation Study Comparison Bar Chart (ablation_study_comparison.png):\n"]
    })
    sec9_outputs.append({
        "data": {
            "image/png": b64_abl,
            "text/plain": ["<IPython.core.display.Image object>"]
        },
        "metadata": {"image/png": {"width": 1050}},
        "output_type": "display_data"
    })

sec9_code = {
    "cell_type": "code",
    "execution_count": 15,
    "metadata": {},
    "outputs": sec9_outputs,
    "source": sec9_code_source
}

# Replace cells 20 and 21 with Section 8 and Section 9
# Keep earlier cells intact (0 to 19)
nb["cells"] = nb["cells"][:20] + [sec8_md, sec8_code, sec9_md, sec9_code]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[OK] Successfully updated 2401CS29_MLProject.ipynb with Section 8 (XAI) and Section 9 (Ablation Study)!")

