#!/usr/bin/env bash
# ==============================================================================
# Build Script for Academic Conference Poster (WD-GPR-SR)
# Author: Moonis Laskar (Roll No: 2401CS29)
# Institution: Indian Institute of Technology Patna (IIT Patna)
# Output: 2401CS29_Project_Poster.pdf (48" x 36" Landscape Conference Poster)
# ==============================================================================

set -e

echo "======================================================================"
echo " STEP 1: Generating High-Resolution 300 DPI Poster Assets"
echo "======================================================================"
python src/generate_poster_charts.py

echo ""
echo "======================================================================"
echo " STEP 2: Compiling Final Academic Conference Poster PDF"
echo "======================================================================"
python src/compile_poster_pdf.py

if [ -f "2401CS29_Project_Poster_fixed.pdf" ]; then
    cp -f "2401CS29_Project_Poster_fixed.pdf" "2401CS29_Project_Poster.pdf" 2>/dev/null || true
fi

echo ""
echo "======================================================================"
echo " [SUCCESS] Conference Poster Successfully Built!"
echo " Deliverables Generated:"
echo "   1. PDF Poster: 2401CS29_Project_Poster_fixed.pdf (TrueType Math Embedded)"
echo "      (also 2401CS29_Project_Poster.pdf once Adobe Acrobat lock is released)"
echo "   2. HTML/CSS Template: docs/poster_template/poster.html"
echo "   3. LaTeX beamerposter: docs/poster_template/poster.tex"
echo "   4. 300 DPI Assets: results/poster_assets/ (5 sub-plots + logo + QR)"
echo "======================================================================"

