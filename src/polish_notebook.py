"""
Polish Jupyter Notebook (2401CS29_MLProject.ipynb) for Final Phase 7 Submission.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction
"""

import json
from pathlib import Path

nb_path = Path("2401CS29_MLProject.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Read Jetson deployment scripts
with open("jetson/export_model.py", "r", encoding="utf-8") as f:
    export_code = f.read()

with open("jetson/benchmark_orin_nano.py", "r", encoding="utf-8") as f:
    benchmark_code = f.read()

# 1. Update Cell 00 (Table of Contents)
new_toc = (
    "## Table of Contents\n\n"
    "1. [Literature Review & Mathematical Formulation](#section-1)\n"
    "2. [Environment Setup & Metric Verification (Day 2)](#day-2-setup)\n"
    "3. [Data Loading and Preprocessing (Days 3–4)](#section-2)\n"
    "4. [Feature Engineering, ICA & LOCO CV Splitting (Days 5–7)](#section-3)\n"
    "5. [Baseline Models & Benchmark Comparison (Days 8–12)](#section-4)\n"
    "6. [Proposed Model & Training: Wide-and-Deep Sparse GPR (Days 13–14)](#section-5)\n"
    "7. [Uncertainty-Weighted Symbolic Regression (Days 15–16)](#section-6)\n"
    "8. [Evaluation: Proposed vs. Baselines Across Seeds (Days 17–18)](#section-7)\n"
    "9. [XAI and Visualization: SHAP, LIME, PDP/ICE, ARD (Days 21–22)](#section-8)\n"
    "10. [Ablation Study: Empirical Architectural Validation (Days 19–20)](#section-9)\n"
    "11. [Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment (Days 23–26)](#section-10)\n"
    "12. [Appendix: Edge Deployment Scripts (`export_model.py` & `benchmark_orin_nano.py`)](#appendix)\n"
)

c0_text = "".join(nb["cells"][0]["source"])
if "## Table of Contents" in c0_text:
    parts = c0_text.split("## Table of Contents")
    updated_c0 = parts[0] + new_toc
    nb["cells"][0]["source"] = [l + "\n" for l in updated_c0.split("\n")[:-1]]

# 2. Refine Specific Section Headings
for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])
    
    # Section 2: Data Loading and Preprocessing
    if "Section 2: NASA Ames" in src:
        new_src = src.replace(
            "Section 2: NASA Ames & CALCE Battery Data Ingestion & Preprocessing",
            "Section 2: Data Loading and Preprocessing (Days 3–4)"
        )
        cell["source"] = [l + "\n" for l in new_src.split("\n")[:-1]]
        
    # Section 5: Proposed Model & Training
    elif "Section 5: Wide-and-Deep Kernel Design" in src:
        new_src = src.replace(
            "Section 5: Wide-and-Deep Kernel Design & SVGP Training",
            "Section 5: Proposed Model & Training: Wide-and-Deep Sparse GPR Architecture (Days 13–14)"
        )
        cell["source"] = [l + "\n" for l in new_src.split("\n")[:-1]]

    # Section 7: Evaluation
    elif "Section 7: Comprehensive Evaluation" in src:
        new_src = src.replace(
            "Section 7: Comprehensive Evaluation & Proposed vs. Baselines Comparison",
            "Section 7: Evaluation: Comprehensive Benchmark & Statistical Analysis (Days 17–18)"
        )
        cell["source"] = [l + "\n" for l in new_src.split("\n")[:-1]]

    # Section 8: XAI and Visualization
    elif "Section 8: Explainable AI" in src:
        new_src = src.replace(
            "Section 8: Explainable AI (XAI) Comprehensive Interpretability Analysis",
            "Section 8: XAI and Visualization: Comprehensive Interpretability Analysis (Days 21–22)"
        )
        cell["source"] = [l + "\n" for l in new_src.split("\n")[:-1]]

    # Section 10: Jetson Measurements
    elif "Section 10: NVIDIA Jetson Orin Nano" in src:
        new_src = src.replace(
            "Section 10: NVIDIA Jetson Orin Nano Edge AI Deployment & Latency Benchmarks",
            "Section 10: Jetson Measurements: NVIDIA Jetson Orin Nano Edge AI Deployment (Days 23–26)"
        )
        cell["source"] = [l + "\n" for l in new_src.split("\n")[:-1]]

# 3. Construct Appendix Markdown Cell
appendix_text = f"""---
# Appendix: Edge Deployment Scripts (`export_model.py` & `benchmark_orin_nano.py`) <a id="appendix"></a>

**Author:** Moonis Laskar (Roll No: `2401CS29`)  
**Project:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  

This appendix documents the complete source code for edge compilation, TensorRT optimization, static C array extraction, and high-resolution hardware benchmarking on the **NVIDIA Jetson Orin Nano**.

---

### Appendix A.1: Model Export & Edge Compilation (`jetson/export_model.py`)
Converts the trained PyTorch WD-GPR-SR model (`B0018`) into an ONNX representation (`models/deep_branch.onnx`), extracts static variational GP matrices (`inducing_points.npy`, `alpha.npy`, `beta_matrix.npy`, `gp_arrays.h`), synthesizes the symbolic residual equation into ANSI C (`symbolic_residual.c`), and compiles the shared library (`symbolic.so` / `symbolic.dll`).

```python
{export_code}
```

---

### Appendix A.2: Edge Hardware Benchmarking Harness (`jetson/benchmark_orin_nano.py`)
Executes 50 DVFS stabilization warm-up loops and 1,000 timed end-to-end inference cycles with nanosecond monotonic timers, profiling median and 95th-percentile latency, peak RAM footprint, power draw, and energy consumption per inference.

```python
{benchmark_code}
```
"""

appendix_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + "\n" for line in appendix_text.split("\n")[:-1]]
}

# Remove old appendix if present
nb["cells"] = [c for c in nb["cells"] if not (c["cell_type"] == "markdown" and any("# Appendix:" in l for l in c["source"]))]

# Append Appendix Cell
nb["cells"].append(appendix_cell)
print(f"Appended Appendix Cell. Total notebook cells: {len(nb['cells'])}")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[OK] Successfully polished {nb_path} with all required sections and Appendix.")

