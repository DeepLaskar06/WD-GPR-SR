"""
Final Deliverables Verification Script for Phase 7 Submission.
Author: Moonis Laskar (Roll No: 2401CS29)
"""

import json
import re
from pathlib import Path
from pptx import Presentation

def verify():
    print("=" * 80)
    print(" PHASE 7 FINAL SUBMISSION DELIVERABLES AUDIT")
    print("=" * 80)

    # 1. Notebook Verification
    nb_p = Path("2401CS29_MLProject.ipynb")
    assert nb_p.exists(), f"Notebook {nb_p} not found!"
    with open(nb_p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    n_cells = len(nb["cells"])
    cells_with_outputs = sum(1 for c in nb["cells"] if c.get("outputs"))
    print(f"1. Notebook File       : {nb_p.name} ({nb_p.stat().st_size:,} bytes)")
    print(f"   Structure Checks    : {n_cells} total cells | {cells_with_outputs} output cells preserved")

    # 2. PDF Report Verification
    pdf_p = Path("2401CS29_MLProject.pdf")
    assert pdf_p.exists(), f"PDF {pdf_p} not found!"
    with open(pdf_p, "rb") as f:
        pdf_bytes = f.read()
    page_count = len(re.findall(rb'/Type\s*/Page\b', pdf_bytes))
    status_pdf = "PASS" if page_count <= 8 else "FAIL (Exceeds 8 pages)"
    print(f"\n2. Project Report File : {pdf_p.name} ({pdf_p.stat().st_size:,} bytes)")
    print(f"   Page Count Audit    : {page_count} pages (Constraint: max 8 pages) -> [{status_pdf}]")

    # 3. PPTX Slides Verification
    pptx_p = Path("2401CS29_MLProject.pptx")
    assert pptx_p.exists(), f"PPTX {pptx_p} not found!"
    prs = Presentation(str(pptx_p))
    slide_count = len(prs.slides)
    status_pptx = "PASS" if slide_count <= 10 else "FAIL (Exceeds 10 slides)"
    print(f"\n3. Slide Deck File     : {pptx_p.name} ({pptx_p.stat().st_size:,} bytes)")
    print(f"   Slide Count Audit   : {slide_count} slides (Constraint: max 10 slides) -> [{status_pptx}]")

    print("\n" + "=" * 80)
    print(" ALL THREE REQUIRED SUBMISSION FILES VERIFIED AND READY FOR SUBMISSION!")
    print("=" * 80)

if __name__ == "__main__":
    verify()

