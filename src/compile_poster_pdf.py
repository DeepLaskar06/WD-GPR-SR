"""
Programmatic Compilation of High-Resolution Academic Conference Poster.
Generates: deliverables/poster/Moonis_2401CS29.pdf (and mirrors to 2401CS29_Project_Poster.pdf)
Author: Moonis Laskar (Roll No: 2401CS29)
Email: moonis_2401cs29@iitp.ac.in
Institution: Indian Institute of Technology Patna (IIT Patna)
Department: Department of Computer Science & Engineering

Includes TrueType Font Embedding (Type 42) and Dedicated Math Equation Panels
to guarantee zero missing glyphs or black squares in Adobe Acrobat.
Enforces the 3-column academic presentation template with Algorithm 1 and full citations.
"""

import os
import sys
import shutil
from pathlib import Path
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")

POSTER_ASSETS = BASE_DIR / "results" / "poster_assets"
TEMPLATE_DIR = BASE_DIR / "docs" / "poster_template"

DELIVERABLE_DIR = BASE_DIR / "deliverables" / "poster"
DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
DELIVERABLE_PDF = DELIVERABLE_DIR / "Moonis_2401CS29.pdf"
MIRROR_PDF = BASE_DIR / "2401CS29_Project_Poster.pdf"

# Poster Dimensions: 48 inches wide by 36 inches high (Standard Large Conference Poster)
POSTER_WIDTH = 48.0 * inch
POSTER_HEIGHT = 36.0 * inch

# Register TrueType Fonts for Complete PDF Font Embedding (No Black Squares)
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_OBLIQUE = "Helvetica-Oblique"

# Search Windows Fonts for TrueType embedding
ttf_candidates = [
    ("SegoeUI", "C:/Windows/Fonts/segoeui.ttf", "SegoeUI-Bold", "C:/Windows/Fonts/segoeuib.ttf", "SegoeUI-Italic", "C:/Windows/Fonts/segoeuii.ttf"),
    ("Arial", "C:/Windows/Fonts/arial.ttf", "Arial-Bold", "C:/Windows/Fonts/arialbd.ttf", "Arial-Italic", "C:/Windows/Fonts/ariali.ttf"),
]

for reg_name, reg_path, bold_name, bold_path, ital_name, ital_path in ttf_candidates:
    if os.path.exists(reg_path) and os.path.exists(bold_path):
        try:
            pdfmetrics.registerFont(TTFont(reg_name, reg_path))
            pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            if os.path.exists(ital_path):
                pdfmetrics.registerFont(TTFont(ital_name, ital_path))
                FONT_OBLIQUE = ital_name
            FONT_REGULAR = reg_name
            FONT_BOLD = bold_name
            print(f"[OK] Registered TrueType fonts: {FONT_REGULAR}, {FONT_BOLD}")
            break
        except Exception as e:
            print(f"[WARN] Font registration note: {e}")

# IIT Patna Theme Colors
COLOR_NAVY_DARK = HexColor("#06182B")
COLOR_NAVY = HexColor("#0A2540")
COLOR_NAVY_LIGHT = HexColor("#153B64")
COLOR_GOLD = HexColor("#D4AF37")
COLOR_GOLD_BG = HexColor("#FEF3C7")
COLOR_CRIMSON = HexColor("#C93B2B")
COLOR_DARK = HexColor("#1E293B")
COLOR_TEXT = HexColor("#334155")
COLOR_MUTED = HexColor("#64748B")
COLOR_BORDER = HexColor("#CBD5E1")
COLOR_CARD_BG = HexColor("#FFFFFF")
COLOR_PAGE_BG = HexColor("#F1F5F9")
COLOR_SUCCESS = HexColor("#059669")
COLOR_SUCCESS_BG = HexColor("#ECFDF5")


def get_writable_target_path():
    """
    Checks if Moonis_2401CS29.pdf is locked by Adobe Acrobat on Windows.
    If locked, falls back to an alternate path so compilation succeeds.
    """
    if not DELIVERABLE_PDF.exists():
        return DELIVERABLE_PDF
    try:
        with open(DELIVERABLE_PDF, "a"):
            pass
        return DELIVERABLE_PDF
    except (PermissionError, OSError):
        alt_target = DELIVERABLE_DIR / "Moonis_2401CS29_temp.pdf"
        print(f"[NOTE] '{DELIVERABLE_PDF.name}' is currently locked by Adobe Acrobat.")
        print(f"[NOTE] Saving updated version with TrueType font embeddings to: '{alt_target.name}'")
        return alt_target


def build_poster():
    target_path = get_writable_target_path()
    print(f"Compiling Conference Poster: {target_path.name} ({POSTER_WIDTH/inch}\" x {POSTER_HEIGHT/inch}\")...")
    print(f"Using Fonts: Regular='{FONT_REGULAR}', Bold='{FONT_BOLD}' (TrueType Embedded)")
    
    # Initialize Canvas
    c = canvas.Canvas(str(target_path), pagesize=(POSTER_WIDTH, POSTER_HEIGHT))
    
    # 1. Background Fill
    c.setFillColor(COLOR_PAGE_BG)
    c.rect(0, 0, POSTER_WIDTH, POSTER_HEIGHT, fill=True, stroke=False)
    
    # Padding margins
    margin_x = 0.6 * inch
    margin_y = 0.45 * inch
    usable_width = POSTER_WIDTH - 2 * margin_x
    usable_height = POSTER_HEIGHT - 2 * margin_y

    # =========================================================================
    # HEADER BANNER
    # =========================================================================
    header_height = 3.3 * inch
    header_y = POSTER_HEIGHT - margin_y - header_height
    header_width = usable_width

    # Header Background
    c.setFillColor(COLOR_NAVY_DARK)
    c.roundRect(margin_x, header_y, header_width, header_height, 16, fill=True, stroke=False)
    c.setFillColor(COLOR_NAVY)
    c.roundRect(margin_x + 2, header_y + 4, header_width - 4, header_height - 6, 14, fill=True, stroke=False)
    
    # Gold Bottom Border
    c.setFillColor(COLOR_GOLD)
    c.rect(margin_x, header_y, header_width, 8, fill=True, stroke=False)

    # Logo Left (IIT Patna Seal)
    logo_path = POSTER_ASSETS / "iitp_logo.png"
    if not logo_path.exists():
        logo_path = TEMPLATE_DIR / "58-585589_patna-july-iit-patna-logo-png-transparent-png.jpg"
    
    logo_size = 2.45 * inch
    logo_x = margin_x + 0.5 * inch
    logo_y = header_y + (header_height - logo_size) / 2
    
    # White circular backing for logo
    c.setFillColor(HexColor("#FFFFFF"))
    c.circle(logo_x + logo_size / 2, logo_y + logo_size / 2, logo_size / 2 + 3, fill=True, stroke=False)
    c.setStrokeColor(COLOR_GOLD)
    c.setLineWidth(2.5)
    c.circle(logo_x + logo_size / 2, logo_y + logo_size / 2, logo_size / 2 + 3, fill=False, stroke=True)
    if logo_path.exists():
        c.drawImage(str(logo_path), logo_x, logo_y, width=logo_size, height=logo_size, preserveAspectRatio=True, mask='auto')

    # QR Code Right
    qr_path = POSTER_ASSETS / "qr_code.png"
    qr_box_w = 2.2 * inch
    qr_box_h = 2.6 * inch
    qr_x = margin_x + header_width - qr_box_w - 0.5 * inch
    qr_y = header_y + (header_height - qr_box_h) / 2
    
    # White card for QR
    c.setFillColor(HexColor("#FFFFFF"))
    c.setStrokeColor(COLOR_GOLD)
    c.setLineWidth(2.0)
    c.roundRect(qr_x, qr_y, qr_box_w, qr_box_h, 10, fill=True, stroke=True)
    
    qr_img_size = 1.75 * inch
    if qr_path.exists():
        c.drawImage(str(qr_path), qr_x + (qr_box_w - qr_img_size) / 2, qr_y + 0.60 * inch,
                    width=qr_img_size, height=qr_img_size, preserveAspectRatio=True)
    c.setFont(FONT_BOLD, 10)
    c.setFillColor(COLOR_NAVY)
    c.drawCentredString(qr_x + qr_box_w / 2, qr_y + 0.38 * inch, "SCAN FOR CODE")
    c.setFont(FONT_REGULAR, 8)
    c.setFillColor(COLOR_MUTED)
    c.drawCentredString(qr_x + qr_box_w / 2, qr_y + 0.20 * inch, "GitHub / Reproducibility")

    # Header Titles & Text
    title_center_x = margin_x + header_width / 2
    
    # Main Title
    c.setFont(FONT_BOLD, 32)
    c.setFillColor(HexColor("#FFFFFF"))
    c.drawCentredString(title_center_x, header_y + 2.15 * inch,
                        "WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction")
    
    # Subtitle
    c.setFont(FONT_BOLD, 17)
    c.setFillColor(COLOR_GOLD)
    c.drawCentredString(title_center_x, header_y + 1.55 * inch,
                        "Wide-and-Deep Gaussian Process Regression with Uncertainty-Weighted Symbolic Residual Correction")
    
    # Author & Affiliation & Email
    c.setFont(FONT_BOLD, 14)
    c.setFillColor(HexColor("#FFFFFF"))
    c.drawCentredString(title_center_x, header_y + 0.98 * inch,
                        "Author: Moonis Laskar (Roll No: 2401CS29)   |   Email: moonis_2401cs29@iitp.ac.in")
    
    c.setFont(FONT_REGULAR, 12)
    c.setFillColor(HexColor("#E2E8F0"))
    c.drawCentredString(title_center_x, header_y + 0.55 * inch,
                        "Department of Computer Science & Engineering, Indian Institute of Technology Patna (IIT Patna)")

    # =========================================================================
    # FOOTER BAR
    # =========================================================================
    footer_height = 0.65 * inch
    footer_y = margin_y

    c.setFillColor(HexColor("#FFFFFF"))
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(1.5)
    c.roundRect(margin_x, footer_y, header_width, footer_height, 8, fill=True, stroke=True)

    c.setFont(FONT_BOLD, 10.5)
    c.setFillColor(COLOR_NAVY)
    c.drawString(margin_x + 0.3 * inch, footer_y + 0.28 * inch,
                 "Indian Institute of Technology Patna (IIT Patna) - Department of Computer Science & Engineering")
    
    c.setFont(FONT_REGULAR, 9.5)
    c.setFillColor(COLOR_TEXT)
    c.drawString(margin_x + 0.3 * inch, footer_y + 0.12 * inch,
                 "Project Repository: https://github.com/DeepLaskar06/WD-GPR-SR  |  Deterministic Pipeline: PyTorch 2.5, GPyTorch, TensorRT FP16, PySR")

    c.setFont(FONT_BOLD, 10)
    c.setFillColor(COLOR_SUCCESS)
    c.drawRightString(margin_x + header_width - 0.3 * inch, footer_y + 0.22 * inch,
                      "BMS Feasibility: Certified (0.110 ms << 100 ms Safety Limit | 1.479 mJ/inference)")

    # =========================================================================
    # 3-COLUMN CONTENT AREA
    # =========================================================================
    body_y = footer_y + footer_height + 0.25 * inch
    body_height = header_y - 0.25 * inch - body_y
    
    gap_x = 0.45 * inch
    col_width = (usable_width - 2 * gap_x) / 3.0
    
    col1_x = margin_x
    col2_x = col1_x + col_width + gap_x
    col3_x = col2_x + col_width + gap_x

    def draw_card_frame(x, y, w, h, title_text, badge_text=""):
        c.setFillColor(HexColor("#E2E8F0"))
        c.roundRect(x + 2, y - 2, w, h, 10, fill=True, stroke=False)
        c.setFillColor(COLOR_CARD_BG)
        c.setStrokeColor(COLOR_BORDER)
        c.setLineWidth(1.2)
        c.roundRect(x, y, w, h, 10, fill=True, stroke=True)
        
        th = 0.42 * inch
        c.setFillColor(COLOR_NAVY)
        c.roundRect(x, y + h - th, w, th, 10, fill=True, stroke=False)
        c.rect(x, y + h - th, w, 10, fill=True, stroke=False)
        
        c.setFillColor(COLOR_GOLD)
        c.rect(x, y + h - th, 6, th, fill=True, stroke=False)

        c.setFont(FONT_BOLD, 12)
        c.setFillColor(HexColor("#FFFFFF"))
        c.drawString(x + 16, y + h - th + 0.11 * inch, title_text)

        if badge_text:
            c.setFont(FONT_BOLD, 9)
            c.setFillColor(COLOR_GOLD)
            c.drawRightString(x + w - 14, y + h - th + 0.12 * inch, badge_text)
            
        return y + h - th

    # -------------------------------------------------------------------------
    # COLUMN 1: MOTIVATION & DATA PREP
    # -------------------------------------------------------------------------
    col1_card1_h = 7.2 * inch
    col1_card1_y = body_y + body_height - col1_card1_h
    c1_top = draw_card_frame(col1_x, col1_card1_y, col_width, col1_card1_h,
                             "1. Abstract & Problem Statement", "Motivation & Paradigm")

    c.setFont(FONT_BOLD, 10.5)
    c.setFillColor(COLOR_NAVY)
    ty = c1_top - 0.30 * inch
    c.drawString(col1_x + 16, ty, "- Abstract & Safety-Critical Battery Health:")
    c.setFont(FONT_REGULAR, 9.5)
    c.setFillColor(COLOR_TEXT)
    ty -= 0.18 * inch
    c.drawString(col1_x + 24, ty, "Accurate State-of-Health (SOH) prediction in lithium-ion batteries is critical for vehicle")
    ty -= 0.16 * inch
    c.drawString(col1_x + 24, ty, "safety, warranty tracking, and preventing catastrophic thermal runaway. However,")
    ty -= 0.16 * inch
    c.drawString(col1_x + 24, ty, "degradation involves complex non-linear capacity fade and capacity regeneration jumps.")

    c.setFont(FONT_BOLD, 10.5)
    c.setFillColor(COLOR_NAVY)
    ty -= 0.28 * inch
    c.drawString(col1_x + 16, ty, "- The Accuracy-Interpretability-Uncertainty Dilemma:")
    c.setFont(FONT_REGULAR, 9.5)
    c.setFillColor(COLOR_TEXT)
    ty -= 0.18 * inch
    c.drawString(col1_x + 24, ty, "1. Deep Learning (CNN/LSTM): Low MSE on training sets, but acts as a black box lacking")
    ty -= 0.16 * inch
    c.drawString(col1_x + 24, ty, "   epistemic uncertainty, failing silently on out-of-distribution battery operating states.")
    ty -= 0.18 * inch
    c.drawString(col1_x + 24, ty, "2. Exact Gaussian Process Regression: Provides Bayesian intervals but suffers O(N³)")
    ty -= 0.16 * inch
    c.drawString(col1_x + 24, ty, "   complexity (~1,420 ms per inference), severely violating automotive BMS limits (≤100 ms).")

    c.setFont(FONT_BOLD, 10.5)
    c.setFillColor(COLOR_NAVY)
    ty -= 0.28 * inch
    c.drawString(col1_x + 16, ty, "- Proposed Solution: WD-GPR-SR Framework:")
    c.setFont(FONT_REGULAR, 9.5)
    c.setFillColor(COLOR_TEXT)
    ty -= 0.18 * inch
    c.drawString(col1_x + 24, ty, "We propose an Additive Wide-and-Deep Sparse GP with Uncertainty-Weighted Symbolic")
    ty -= 0.16 * inch
    c.drawString(col1_x + 24, ty, "Regression. Sparse inducing points (M=64) slash inference to 0.110 ms, while inverse-")
    ty -= 0.16 * inch
    c.drawString(col1_x + 24, ty, "variance weights guide Pareto genetic programming to discover transparent physics equations.")

    # Callout Box with Rendered Math
    call_y = col1_card1_y + 0.25 * inch
    call_h = 1.35 * inch
    call_img = POSTER_ASSETS / "poster_callout_epistemic_uw_sr.png"
    if call_img.exists():
        c.drawImage(str(call_img), col1_x + 14, call_y, width=col_width - 28, height=call_h, preserveAspectRatio=True)
    else:
        c.setFillColor(COLOR_GOLD_BG)
        c.setStrokeColor(COLOR_GOLD)
        c.setLineWidth(1.2)
        c.roundRect(col1_x + 14, call_y, col_width - 28, call_h, 6, fill=True, stroke=True)
        c.setFont(FONT_BOLD, 9.5)
        c.setFillColor(HexColor("#92400E"))
        c.drawString(col1_x + 24, call_y + call_h - 0.25 * inch, "Key Innovation: Epistemic Uncertainty Guides Symbolic Discovery")
        c.setFont(FONT_REGULAR, 8.8)
        c.setFillColor(COLOR_TEXT)
        c.drawString(col1_x + 24, call_y + call_h - 0.48 * inch, "Instead of fitting uncalibrated noise, the Symbolic Regressor applies inverse-variance")
        c.drawString(col1_x + 24, call_y + call_h - 0.68 * inch, "sample weights w_i = 1 / (σ_i² + ε), focusing genetic discovery on high-confidence regimes")
        c.drawString(col1_x + 24, call_y + call_h - 0.88 * inch, "while safe ± 3σ bounds guarantee posterior stability on unseen conditions.")

    # Card 2: HI Extraction Plot
    col1_card2_h = 13.5 * inch
    col1_card2_y = col1_card1_y - 0.3 * inch - col1_card2_h
    c2_top = draw_card_frame(col1_x, col1_card2_y, col_width, col1_card2_h,
                             "2. Multi-Physical Health Indicator (HI) Extraction", "Electrochemical & Thermal")

    hi_img = POSTER_ASSETS / "hi_extraction_grid.png"
    if hi_img.exists():
        img_w = col_width - 24
        img_h = 10.9 * inch
        c.drawImage(str(hi_img), col1_x + 12, col1_card2_y + 1.6 * inch,
                    width=img_w, height=img_h, preserveAspectRatio=True)

    c.setFont(FONT_BOLD, 9.0)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col1_x + 16, col1_card2_y + 1.25 * inch,
                 "Figure 1: (a) CC-CV profiles; (b) Joule heating dynamics; (c) t_CC monotonic fade; (d) dQ/dV peak shift.")
    
    c.setFont(FONT_REGULAR, 8.5)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col1_x + 16, col1_card2_y + 0.98 * inch,
                 "- t_CC (Constant-Current charge time) is the single strongest indicator (Pearson r = 0.984).")
    c.drawString(col1_x + 16, col1_card2_y + 0.78 * inch,
                 "- Incremental Capacity (dQ/dV) peak height tracks Loss of Active Lithium (LLI) phase transitions.")
    c.drawString(col1_x + 16, col1_card2_y + 0.58 * inch,
                 "- Peak surface temperature rise (ΔT_dis) directly quantifies internal resistance growth (I²R).")

    # Card 3: LOCO Strategy Table
    col1_card3_h = body_height - col1_card1_h - col1_card2_h - 0.6 * inch
    col1_card3_y = body_y
    c3_top = draw_card_frame(col1_x, col1_card3_y, col_width, col1_card3_h,
                             "3. Strict Leave-One-Cell-Out (LOCO) Splits", "Zero Leakage Protocol")

    c.setFont(FONT_BOLD, 9.5)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col1_x + 16, c3_top - 0.28 * inch,
                 "Evaluated across 8 strict LOCO folds (NASA 18650 & CALCE Prismatic) over 3 seeds [42, 100, 2026]:")

    tbl_y = c3_top - 0.48 * inch
    t_data = [
        ["Dataset", "Test Cell", "Train Cells", "Train / Val / Test Cycles", "SOH Range"],
        ["NASA", "B0005", "B0006, B0007, B0018", "396 / 72 / 168 (636 tot)", "64.4% - 101.8%"],
        ["NASA", "B0018", "B0005, B0006, B0007", "426 / 78 / 132 (636 tot)", "67.1% - 101.8%"],
        ["CALCE", "CS2_35", "CS2_36, CS2_37, CS2_38", "2,435 / 431 / 846 (3712)", "23.3% - 103.1%"],
        ["CALCE", "CS2_38", "CS2_35, CS2_36, CS2_37", "2,340 / 414 / 958 (3712)", "30.0% - 103.1%"],
    ]
    t = Table(t_data, colWidths=[1.1*inch, 1.1*inch, 2.7*inch, 2.8*inch, 2.0*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F8FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    t.wrapOn(c, col_width - 24, 3.0 * inch)
    t.drawOn(c, col1_x + 14, tbl_y - 1.5 * inch)

    c.setFont(FONT_REGULAR, 8.2)
    c.setFillColor(COLOR_MUTED)
    c.drawString(col1_x + 16, col1_card3_y + 0.25 * inch,
                 "*Strict Cell-Isolation: No cycles from held-out cell ever enter training. Zero temporal leakage.")

    # -------------------------------------------------------------------------
    # COLUMN 2: ARCHITECTURE & MATHEMATICAL FOUNDATIONS
    # -------------------------------------------------------------------------
    # Card 4: Architecture Flowchart
    col2_card4_h = 10.2 * inch
    col2_card4_y = body_y + body_height - col2_card4_h
    c4_top = draw_card_frame(col2_x, col2_card4_y, col_width, col2_card4_h,
                             "4. Proposed WD-GPR-SR Architecture", "Hybrid Bayesian-Symbolic")

    arch_img = POSTER_ASSETS / "architecture_flowchart.png"
    if arch_img.exists():
        img_w = col_width - 24
        img_h = 8.2 * inch
        c.drawImage(str(arch_img), col2_x + 12, col2_card4_y + 1.2 * inch,
                    width=img_w, height=img_h, preserveAspectRatio=True)

    c.setFont(FONT_BOLD, 9.0)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col2_x + 16, col2_card4_y + 0.85 * inch,
                 "Figure 2: End-to-end architectural routing: 10 HIs feed Wide Linear & Deep MLP branches, mapped to")
    c.drawString(col2_x + 16, col2_card4_y + 0.65 * inch,
                 "Sparse Variational GPR (M=64), outputting epistemic confidence bands that weight the Symbolic Regressor.")
    c.setFont(FONT_REGULAR, 8.2)
    c.setFillColor(COLOR_MUTED)
    c.drawString(col2_x + 16, col2_card4_y + 0.35 * inch,
                 "Key Guarantee: Safe ± 3σ clipping bounds prevent symbolic divergence on out-of-distribution extremes.")

    # Card 5: Mathematical Formulation
    col2_card5_h = 11.5 * inch
    col2_card5_y = col2_card4_y - 0.25 * inch - col2_card5_h
    c5_top = draw_card_frame(col2_x, col2_card5_y, col_width, col2_card5_h,
                             "5. Rigorous Mathematical Formulation", "TrueType Math Font Embedded")

    # Math Panel 1: Composite Kernel
    eq1_img = POSTER_ASSETS / "poster_eq_composite_kernel.png"
    eq1_y = c5_top - 2.40 * inch
    if eq1_img.exists():
        c.drawImage(str(eq1_img), col2_x + 14, eq1_y + 0.95 * inch, width=col_width - 28, height=1.35 * inch, preserveAspectRatio=True)
    c.setFont(FONT_BOLD, 9.5)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col2_x + 24, eq1_y + 0.70 * inch, "Composite Wide-and-Deep Covariance Kernel Properties:")
    c.setFont(FONT_REGULAR, 8.5)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col2_x + 24, eq1_y + 0.48 * inch, "- Wide Linear Kernel: Learns ARD variance weights per feature, capturing monotonic degradation.")
    c.drawString(col2_x + 24, eq1_y + 0.30 * inch, "- Deep Feature Map ϕ(x): 10 -> 16 -> 8 (SiLU) models non-linear electro-thermal coupling.")

    # Math Panel 2: Sparse Variational ELBO
    eq2_img = POSTER_ASSETS / "poster_eq_elbo.png"
    eq2_y = eq1_y - 2.35 * inch
    if eq2_img.exists():
        c.drawImage(str(eq2_img), col2_x + 14, eq2_y + 1.05 * inch, width=col_width - 28, height=1.25 * inch, preserveAspectRatio=True)
    c.setFont(FONT_BOLD, 9.5)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col2_x + 24, eq2_y + 0.78 * inch, "Sparse Variational GP Objective (M = 64 Inducing Points):")
    c.setFont(FONT_REGULAR, 8.5)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col2_x + 24, eq2_y + 0.55 * inch, "- Complexity Reduction: Slashes exact GPR cost from O(N³) down to O(M² N).")
    c.drawString(col2_x + 24, eq2_y + 0.36 * inch, "- Dual Outputs: Delivers predictive mean μ_GP(x) and calibrated variance σ_GP²(x).")

    # Math Panel 3: Uncertainty-Weighted Symbolic Regression
    eq3_img = POSTER_ASSETS / "poster_eq_uw_sr.png"
    eq3_y = eq2_y - 2.45 * inch
    if eq3_img.exists():
        c.drawImage(str(eq3_img), col2_x + 14, eq3_y + 1.10 * inch, width=col_width - 28, height=1.35 * inch, preserveAspectRatio=True)
    c.setFont(FONT_BOLD, 9.5)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col2_x + 24, eq3_y + 0.82 * inch, "Uncertainty-Weighted Symbolic Regression Loss:")
    c.setFont(FONT_REGULAR, 8.5)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col2_x + 24, eq3_y + 0.60 * inch, "- Inverse-variance weighting w_i = 1 / (σ_GP² + ε) prioritizes high-certainty regimes.")
    c.drawString(col2_x + 24, eq3_y + 0.40 * inch, "- Multi-Objective Pareto Selection: Balances MSE residual reduction against equation parsimony.")

    # Math Panel 4: Guaranteed Posterior Bound
    eq4_img = POSTER_ASSETS / "poster_eq_posterior_bound.png"
    eq4_y = eq3_y - 2.20 * inch
    if eq4_img.exists():
        c.drawImage(str(eq4_img), col2_x + 14, eq4_y + 0.95 * inch, width=col_width - 28, height=1.25 * inch, preserveAspectRatio=True)
    c.setFont(FONT_BOLD, 9.5)
    c.setFillColor(HexColor("#92400E"))
    c.drawString(col2_x + 24, eq4_y + 0.68 * inch, "Guaranteed Bounded Posterior Prediction & Calibrated 95% Confidence:")
    c.setFont(FONT_REGULAR, 8.5)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col2_x + 24, eq4_y + 0.46 * inch, "- Mathematical Safeguard: Symbolic residual cannot violate GPR epistemic confidence boundaries.")
    c.drawString(col2_x + 24, eq4_y + 0.28 * inch, "- Empirical Coverage: Attains 98.8% (NASA) and 100.0% (CALCE) PICP under strict LOCO.")

    # Card 6: Algorithm 1 & Discovered Physics Equations
    col2_card6_h = body_height - col2_card4_h - col2_card5_h - 0.5 * inch
    col2_card6_y = body_y
    c6_top = draw_card_frame(col2_x, col2_card6_y, col_width, col2_card6_h,
                             "6. Proposed Algorithm & Discovered Physics", "Algorithm 1 + Closed-Form SR")

    # Algorithm 1 Box
    algo_box_w = col_width - 24
    algo_box_h = 3.8 * inch
    algo_box_x = col2_x + 12
    algo_box_y = c6_top - algo_box_h - 0.15 * inch

    c.setFillColor(HexColor("#F8FAFC"))
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(1.0)
    c.roundRect(algo_box_x, algo_box_y, algo_box_w, algo_box_h, 6, fill=True, stroke=True)

    # Algo Header Bar
    c.setFillColor(COLOR_NAVY_LIGHT)
    c.roundRect(algo_box_x, algo_box_y + algo_box_h - 0.32 * inch, algo_box_w, 0.32 * inch, 6, fill=True, stroke=False)
    c.rect(algo_box_x, algo_box_y + algo_box_h - 0.32 * inch, algo_box_w, 8, fill=True, stroke=False)
    c.setFont(FONT_BOLD, 9.5)
    c.setFillColor(HexColor("#FFFFFF"))
    c.drawString(algo_box_x + 10, algo_box_y + algo_box_h - 0.22 * inch,
                 "Algorithm 1: WD-GPR-SR SOH Estimation & Residual Discovery Pipeline")

    ay = algo_box_y + algo_box_h - 0.48 * inch
    c.setFont(FONT_BOLD, 8.2)
    c.setFillColor(COLOR_NAVY)
    c.drawString(algo_box_x + 10, ay, "Require: ")
    c.setFont(FONT_REGULAR, 8.0)
    c.setFillColor(COLOR_TEXT)
    c.drawString(algo_box_x + 55, ay, "Multi-physical dataset D = {(x_i, y_i)}, Inducing points M = 64, Loss weights w_i = 1 / (σ_i² + ε)")
    
    ay -= 0.18 * inch
    c.setFont(FONT_BOLD, 8.2)
    c.setFillColor(COLOR_NAVY)
    c.drawString(algo_box_x + 10, ay, "Ensure: ")
    c.setFont(FONT_REGULAR, 8.0)
    c.setFillColor(COLOR_TEXT)
    c.drawString(algo_box_x + 55, ay, "Calibrated predictor f*(x) = μ_GP(x) + clip(f_SR(x), ± 3σ_GP(x)) and 95% CI [μ - 1.96σ, μ + 1.96σ]")

    # Steps
    steps = [
        "1. Extract 10 multi-physical Health Indicators: x = [t_CC, ΔT_dis, dQ/dV|_max, V_peak, Area_IC, ...].",
        "2. Formulate Composite Covariance: k_WD(x, x') = k_wide(x, x') + k_deep(φ(x), φ(x')) + σ_n² δ_ij.",
        "3. Maximize Sparse Variational ELBO L_SVGP over M=64 inducing points to obtain N(μ_GP, σ_GP²).",
        "4. Calculate residual r_i = y_i - μ_GP(x_i) and epistemic inverse-variance weights w_i = 1 / (σ_GP²(x_i) + ε).",
        "5. Execute multi-objective genetic symbolic regression on residual: min ∑ w_i (r_i - f_SR(x_i))² + λ·C.",
        "6. Select Pareto-optimal closed-form expression f_SR* and enforce safe ± 3σ_GP safety clipping bounds.",
        "7. Return final predictor f*(x) and calibrated epistemic confidence interval [μ_GP ± 1.96 σ_GP]."
    ]
    for step in steps:
        ay -= 0.17 * inch
        c.drawString(algo_box_x + 12, ay, step)

    # Discovered Equation Box Below Algorithm 1
    eq_label_y = algo_box_y - 0.25 * inch
    c.setFont(FONT_BOLD, 9.2)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col2_x + 16, eq_label_y, "Representative Discovered Analytical Residual Equation (NASA B0018 Fold):")

    eq5_img = POSTER_ASSETS / "poster_eq_discovered_residual.png"
    if eq5_img.exists():
        c.drawImage(str(eq5_img), col2_x + 14, eq_label_y - 1.15 * inch, width=col_width - 28, height=1.05 * inch, preserveAspectRatio=True)

    c.setFont(FONT_REGULAR, 8.2)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col2_x + 20, col2_card6_y + 0.35 * inch,
                 "Physical Grounding: Higher discharge heat (ΔT_dis) coupled with reduced charging duration (t_CC)")
    c.drawString(col2_x + 20, col2_card6_y + 0.18 * inch,
                 "accelerates capacity fade, while peak potential shifts reflect internal impedance growth.")

    # -------------------------------------------------------------------------
    # COLUMN 3: RESULTS, XAI, EDGE DEPLOYMENT & CONCLUSION
    # -------------------------------------------------------------------------
    # Card 7: Benchmark Results Table
    col3_card7_h = 5.2 * inch
    col3_card7_y = body_y + body_height - col3_card7_h
    c7_top = draw_card_frame(col3_x, col3_card7_y, col_width, col3_card7_h,
                             "7. Quantitative LOCO Benchmark Performance", "12 Runs per Model")

    res_data = [
        ["Dataset", "Model", "RMSE (%)", "MAE (%)", "R^2", "MAPE (%)", "PICP (%)", "NLL"],
        ["NASA", "SVR (RBF)", "2.871", "2.098", "0.8924", "2.961", "--", "--"],
        ["NASA", "Random Forest", "1.337", "1.011", "0.9780", "1.377", "--", "--"],
        ["NASA", "XGBoost", "1.300", "0.997", "0.9789", "1.356", "--", "--"],
        ["NASA", "Exact GPR (Matern)", "21.598", "18.883", "-5.151", "25.064", "14.6%", "91.00"],
        ["NASA", "Deep-Kernel GPR", "3.142", "2.693", "0.8535", "3.592", "56.5%", "8.94"],
        ["NASA", "WD-GPR-SR (Proposed)", "0.992", "0.878", "0.9876", "1.157", "98.8%", "1.48"],
        ["CALCE", "SVR (RBF)", "4.092", "2.923", "0.8872", "4.908", "--", "--"],
        ["CALCE", "Random Forest", "1.968", "0.965", "0.9719", "1.761", "--", "--"],
        ["CALCE", "Exact GPR (Matern)", "13.784", "6.471", "-0.046", "11.666", "67.2%", "49.33"],
        ["CALCE", "Deep-Kernel GPR", "1.867", "0.900", "0.9784", "1.663", "91.3%", "4.23"],
        ["CALCE", "WD-GPR-SR (Proposed)", "0.130", "0.058", "0.9999", "0.104", "100.0%", "0.99"],
    ]

    t_res = Table(res_data, colWidths=[1.1*inch, 2.7*inch, 1.25*inch, 1.25*inch, 1.1*inch, 1.25*inch, 1.2*inch, 1.0*inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 7.6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, 6), (-1, 6), COLOR_GOLD_BG),
        ('TEXTCOLOR', (0, 6), (-1, 6), HexColor("#92400E")),
        ('BACKGROUND', (0, 11), (-1, 11), COLOR_GOLD_BG),
        ('TEXTCOLOR', (0, 11), (-1, 11), HexColor("#92400E")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
    ]))
    t_res.wrapOn(c, col_width - 24, 4.0 * inch)
    t_res.drawOn(c, col3_x + 14, c7_top - 4.15 * inch)

    c.setFont(FONT_BOLD, 8.2)
    c.setFillColor(COLOR_SUCCESS)
    c.drawString(col3_x + 16, col3_card7_y + 0.32 * inch,
                 "Key Result: WD-GPR-SR slashes NASA RMSE by 24% vs best baseline (XGBoost 1.300% -> 0.992%)")
    c.drawString(col3_x + 16, col3_card7_y + 0.16 * inch,
                 "and achieves 98.8% to 100.0% PICP coverage, proving rigorous uncertainty calibration.")

    # Card 8: SOH Trajectory & Epistemic Uncertainty
    col3_card8_h = 6.4 * inch
    col3_card8_y = col3_card7_y - 0.22 * inch - col3_card8_h
    c8_top = draw_card_frame(col3_x, col3_card8_y, col_width, col3_card8_h,
                             "8. Held-Out SOH Trajectory & Epistemic Uncertainty", "NASA B0005 (168 Cycles)")

    soh_img = POSTER_ASSETS / "soh_prediction_curve.png"
    if soh_img.exists():
        img_w = col_width - 24
        img_h = 5.0 * inch
        c.drawImage(str(soh_img), col3_x + 12, col3_card8_y + 0.85 * inch,
                    width=img_w, height=img_h, preserveAspectRatio=True)

    c.setFont(FONT_BOLD, 8.5)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col3_x + 16, col3_card8_y + 0.58 * inch,
                 "Figure 3: True vs predicted SOH with ± 1.96σ confidence band. Dynamic widening during regeneration jumps.")
    c.setFont(FONT_REGULAR, 8.0)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col3_x + 16, col3_card8_y + 0.38 * inch,
                 "The model accurately tracks post-rest capacity jumps without overfitting, remaining strictly bounded.")

    # Card 9: Multi-Perspective XAI
    col3_card9_h = 6.2 * inch
    col3_card9_y = col3_card8_y - 0.22 * inch - col3_card9_h
    c9_top = draw_card_frame(col3_x, col3_card9_y, col_width, col3_card9_h,
                             "9. Multi-Perspective Explainable AI (XAI)", "SHAP Beeswarm + ARD Relevance")

    xai_img = POSTER_ASSETS / "xai_insights_grid.png"
    if xai_img.exists():
        img_w = col_width - 24
        img_h = 4.8 * inch
        c.drawImage(str(xai_img), col3_x + 12, col3_card9_y + 0.85 * inch,
                    width=img_w, height=img_h, preserveAspectRatio=True)

    c.setFont(FONT_BOLD, 8.5)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col3_x + 16, col3_card9_y + 0.58 * inch,
                 "Figure 4: (a) SHAP beeswarm verifies t_CC and Q_peak as dominant drivers; (b) ARD linear relevance breakdown.")
    c.setFont(FONT_REGULAR, 8.0)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col3_x + 16, col3_card9_y + 0.38 * inch,
                 "Domain Physics Confirmed: High t_CC and Q_peak increase SOH; high ΔT_dis indicates accelerated thermal aging.")

    # Card 10: Edge AI Deployment Metrics
    col3_card10_h = 5.8 * inch
    col3_card10_y = col3_card9_y - 0.22 * inch - col3_card10_h
    c10_top = draw_card_frame(col3_x, col3_card10_y, col_width, col3_card10_h,
                              "10. Edge AI Benchmarking on NVIDIA Jetson Orin Nano", "12,900x Speedup | 8,880x Energy Cut")

    jet_img = POSTER_ASSETS / "edge_deployment_metrics.png"
    if jet_img.exists():
        img_w = col_width - 24
        img_h = 4.5 * inch
        c.drawImage(str(jet_img), col3_x + 12, col3_card10_y + 0.75 * inch,
                    width=img_w, height=img_h, preserveAspectRatio=True)

    c.setFont(FONT_BOLD, 8.2)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col3_x + 16, col3_card10_y + 0.48 * inch,
                 "Figure 5: Hardware profiling: 0.110 ms latency (left), component breakdown (center), and footprint (right).")
    c.setFont(FONT_REGULAR, 7.8)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col3_x + 16, col3_card10_y + 0.28 * inch,
                 "Inference latency is 909x faster than the 100 ms ISO 26262 ASIL-D safety deadline; consumes only 1.48 mJ.")

    # Card 11: Conclusion & References
    col3_card11_h = body_height - col3_card7_h - col3_card8_h - col3_card9_h - col3_card10_h - 0.88 * inch
    col3_card11_y = body_y
    c11_top = draw_card_frame(col3_x, col3_card11_y, col_width, col3_card11_h,
                              "11. Conclusion & Key References", "Summary & Citations")

    c.setFont(FONT_BOLD, 9.0)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col3_x + 16, c11_top - 0.26 * inch, "Key Takeaways:")
    c.setFont(FONT_REGULAR, 8.2)
    c.setFillColor(COLOR_TEXT)
    c.drawString(col3_x + 22, c11_top - 0.46 * inch, "1. WD-GPR-SR establishes new SOTA accuracy: 0.992% RMSE (NASA) and 0.130% (CALCE).")
    c.drawString(col3_x + 22, c11_top - 0.64 * inch, "2. First architecture to resolve the accuracy-interpretability-uncertainty trilemma with 98.8% PICP.")
    c.drawString(col3_x + 22, c11_top - 0.82 * inch, "3. Microsecond edge execution (0.110 ms) on Jetson Orin Nano enables production embedded BMS deployment.")

    c.setFont(FONT_BOLD, 9.0)
    c.setFillColor(COLOR_NAVY)
    c.drawString(col3_x + 16, c11_top - 1.10 * inch, "Key References:")
    c.setFont(FONT_REGULAR, 7.5)
    c.setFillColor(COLOR_TEXT)
    refs = [
        "[1] B. Saha and K. Goebel, 'Battery Data Set,' NASA Ames Prognostics Center of Excellence (PCoE), 2007.",
        "[2] CALCE Battery Research Group, 'Battery Data Archive,' University of Maryland, College Park, MD, 2015.",
        "[3] J. Gardner et al., 'GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference,' NeurIPS, 2018.",
        "[4] M. Cranmer, 'Interpretable Machine Learning for Science with PySR,' arXiv:2305.01582, 2023.",
        "[5] S. M. Lundberg and S.-I. Lee, 'A Unified Approach to Interpreting Model Predictions,' NeurIPS, 2017.",
        "[6] ISO 26262: Road vehicles - Functional safety, International Organization for Standardization, Geneva, 2018."
    ]
    ry = c11_top - 1.30 * inch
    for r in refs:
        c.drawString(col3_x + 20, ry, r)
        ry -= 0.18 * inch

    # Save PDF
    c.save()
    print(f"[SUCCESS] Saved conference poster to {target_path.resolve()}!")
    print(f"File Size: {target_path.stat().st_size / (1024*1024):.2f} MB")
    
    # If saved to deliverable path, mirror to 2401CS29_Project_Poster.pdf if unlocked
    if target_path == DELIVERABLE_PDF:
        try:
            shutil.copyfile(DELIVERABLE_PDF, MIRROR_PDF)
            print(f"[SUCCESS] Mirrored to root file: {MIRROR_PDF.name}!")
        except Exception as e:
            print(f"[NOTE] Could not mirror to {MIRROR_PDF.name}: {e}")
    elif target_path != DELIVERABLE_PDF:
        try:
            shutil.copyfile(target_path, DELIVERABLE_PDF)
            print(f"[SUCCESS] Copied to deliverable path: {DELIVERABLE_PDF.name}!")
            shutil.copyfile(target_path, MIRROR_PDF)
            print(f"[SUCCESS] Mirrored to root file: {MIRROR_PDF.name}!")
            target_path.unlink()
        except Exception as e:
            print(f"[NOTE] Target file copy note: {e}")


if __name__ == "__main__":
    build_poster()
