"""
Clean Notebook Schedule References for Final Submission.
Author: Moonis Laskar (Roll No: 2401CS29)
Replaces schedule/day-based headers with clean, professional, fully-implemented section titles.
"""

import json
import re
from pathlib import Path

nb_path = Path("2401CS29_MLProject.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 00 TOC
new_toc = (
    "## Table of Contents\n\n"
    "1. [Literature Review & Mathematical Formulation](#section-1)\n"
    "2. [Environment Setup, Configuration & Metric Verification](#day-2-setup)\n"
    "3. [Data Loading and Preprocessing](#section-2)\n"
    "4. [Feature Engineering, ICA & LOCO CV Splitting](#section-3)\n"
    "5. [Baseline Models & Comprehensive Benchmark Comparison](#section-4)\n"
    "6. [Proposed Model & Training: Wide-and-Deep Sparse GPR Architecture](#section-5)\n"
    "7. [Uncertainty-Weighted Symbolic Regression](#section-6)\n"
    "8. [Evaluation: Comprehensive Benchmark & Statistical Analysis](#section-7)\n"
    "9. [XAI and Visualization: Comprehensive Interpretability Analysis](#section-8)\n"
    "10. [Ablation Study: Empirical Architectural Validation](#section-9)\n"
    "11. [Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment](#section-10)\n"
    "12. [Appendix: Edge Deployment Scripts (`export_model.py` & `benchmark_orin_nano.py`)](#appendix)\n"
)

c0_text = "".join(nb["cells"][0]["source"])
if "## Table of Contents" in c0_text:
    parts = c0_text.split("## Table of Contents")
    updated_c0 = parts[0] + new_toc
    nb["cells"][0]["source"] = [l + "\n" for l in updated_c0.split("\n")[:-1]]

# 2. Iterate and Clean All Cells
replacements = [
    # Cell 02
    ("# Day 2: Environment Setup, Configuration & Metric Verification", "# Section 1.5: Environment Setup, Configuration & Metric Verification"),
    
    # Cell 04
    ("# Section 2: Data Loading and Preprocessing (Days 3–4)", "# Section 2: Data Loading and Preprocessing"),
    
    # Cell 08
    ("7 Day 5 Time/Thermal features", "7 Time/Thermal features extracted in Section 3.1"),
    ("from Day 5", "from Section 3.1"),
    
    # Cell 12
    ("# Section 4: Baseline Models & Comprehensive Benchmark Comparison (Days 8–12)", "# Section 4: Baseline Models & Comprehensive Benchmark Comparison"),
    ("(Days 8–9)", ""),
    ("(Day 10)", ""),
    ("(Days 11–12)", ""),
    
    # Cell 14
    ("# Section 5: Proposed Model & Training: Wide-and-Deep Sparse GPR Architecture (Days 13–14) (Days 13–14)", "# Section 5: Proposed Model & Training: Wide-and-Deep Sparse GPR Architecture"),
    ("# Section 5: Proposed Model & Training: Wide-and-Deep Sparse GPR Architecture (Days 13–14)", "# Section 5: Proposed Model & Training: Wide-and-Deep Sparse GPR Architecture"),
    ("# Section 5: Wide-and-Deep Kernel Design & SVGP Training (Days 13–14)", "# Section 5: Proposed Model & Training: Wide-and-Deep Sparse GPR Architecture"),
    
    # Cell 16
    ("# Section 6: Uncertainty-Weighted Symbolic Regression (Days 15–16)", "# Section 6: Uncertainty-Weighted Symbolic Regression"),
    
    # Cell 18
    ("# Section 7: Evaluation: Comprehensive Benchmark & Statistical Analysis (Days 17–18) (Days 17–18)", "# Section 7: Evaluation: Comprehensive Benchmark & Statistical Analysis"),
    ("# Section 7: Evaluation: Comprehensive Benchmark & Statistical Analysis (Days 17–18)", "# Section 7: Evaluation: Comprehensive Benchmark & Statistical Analysis"),
    ("# Section 7: Comprehensive Evaluation & Proposed vs. Baselines Comparison (Days 17–18)", "# Section 7: Evaluation: Comprehensive Benchmark & Statistical Analysis"),
    
    # Cell 20
    ("# Section 8: XAI and Visualization: Comprehensive Interpretability Analysis (Days 21–22) (Days 21–22)", "# Section 8: XAI and Visualization: Comprehensive Interpretability Analysis"),
    ("# Section 8: XAI and Visualization: Comprehensive Interpretability Analysis (Days 21–22)", "# Section 8: XAI and Visualization: Comprehensive Interpretability Analysis"),
    ("# Section 8: Explainable AI (XAI) Comprehensive Interpretability Analysis (Days 21–22)", "# Section 8: XAI and Visualization: Comprehensive Interpretability Analysis"),
    
    # Cell 22
    ("# Section 9: Ablation Study & Empirical Architectural Validation (Days 19–20)", "# Section 9: Ablation Study & Empirical Architectural Validation"),
    
    # Cell 24
    ("# Section 10: Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment (Days 23–26) (Days 23–26)", "# Section 10: Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment"),
    ("# Section 10: Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment (Days 23–26)", "# Section 10: Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment"),
    ("# Section 10: NVIDIA Jetson Orin Nano Edge AI Deployment & Latency Benchmarks (Days 23–26)", "# Section 10: Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment"),
    ("4. **Rigorous Hardware Benchmarking Protocol (Days 25–26):**", "4. **Rigorous Hardware Benchmarking Protocol:**"),
]

for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])
    for old_val, new_val in replacements:
        if old_val in src:
            src = src.replace(old_val, new_val)
    cell["source"] = [l + "\n" for l in src.split("\n")[:-1]]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[OK] Successfully cleaned all schedule and day-based headers in {nb_path}.")
