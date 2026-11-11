@echo off
REM ==============================================================================
REM Build Script for Academic Conference Poster (WD-GPR-SR) - Windows Batch
REM Author: Moonis Laskar (Roll No: 2401CS29)
REM Institution: Indian Institute of Technology Patna (IIT Patna)
REM Output: 2401CS29_Project_Poster.pdf (48" x 36" Landscape Conference Poster)
REM ==============================================================================

echo ======================================================================
echo  STEP 1: Generating High-Resolution 300 DPI Poster Assets
echo ======================================================================
python src\generate_poster_charts.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to generate poster charts.
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo  STEP 2: Compiling Final Academic Conference Poster PDF
echo ======================================================================
python src\compile_poster_pdf.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile poster PDF.
    exit /b %ERRORLEVEL%
)

REM If original file was locked during compile, try copying once more
if exist 2401CS29_Project_Poster_fixed.pdf (
    copy /y 2401CS29_Project_Poster_fixed.pdf 2401CS29_Project_Poster.pdf >nul 2>&1
)

echo.
echo ======================================================================
echo  [SUCCESS] Conference Poster Successfully Built!
echo  Deliverables Generated:
echo    1. PDF Poster: 2401CS29_Project_Poster_fixed.pdf (TrueType Math Embedded)
echo       (also 2401CS29_Project_Poster.pdf once Adobe Acrobat lock is released)
echo    2. HTML/CSS Template: docs\poster_template\poster.html
echo    3. LaTeX beamerposter: docs\poster_template\poster.tex
echo    4. 300 DPI Assets: results\poster_assets\ (5 sub-plots + logo + QR)
echo ======================================================================

