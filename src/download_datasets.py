"""
Automated Dataset Downloader, Citation Provider, and Raw File Integrity Inspector.
Author: Moonis Laskar (Roll No: 2401CS29)
Project: WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
import scipy.io
from tqdm import tqdm

from src.config import (
    DATA_RAW_NASA_DIR,
    DATA_RAW_CALCE_DIR,
    NASA_CELLS,
    NASA_NOMINAL_CAP,
    CALCE_CELLS,
    CALCE_NOMINAL_CAP,
)

# =====================================================================
# 1. Dataset Citations & Remote Archive Metadata
# =====================================================================

NASA_CITATION = """
==================================================================================
 NASA Prognostics Center of Excellence (PCoE) Battery Dataset Citation
==================================================================================
 Citation:
   B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data
   Repository, NASA Ames Research Center, Moffett Field, CA.
   URL: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
   Description:
     Commercial 18650 LiCoO2 cells (nominal 2.0 Ah) cycled at room temperature (24°C)
     through operational charge, discharge, and impedance measurement regimes.
==================================================================================
"""

CALCE_CITATION = """
==================================================================================
 CALCE Battery Data Archive (University of Maryland) Citation
==================================================================================
 Citation:
   Center for Advanced Life Cycle Engineering (CALCE) Battery Research Group,
   University of Maryland, College Park, MD, USA.
   M. Pecht et al. "Lithium-ion Battery Aging Datasets under Dynamic Cycling Conditions".
   URL: https://calce.umd.edu/battery-data / https://web.calce.umd.edu/batteries/data.htm
   Description:
     Prismatic commercial LiCoO2 cells (nominal 1.1 Ah) cycled with Arbin battery
     testing systems under constant-current / constant-voltage protocols.
==================================================================================
"""

# Remote Mirrors
NASA_MIRRORS = {
    cell: [
        f"https://raw.githubusercontent.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset/master/{cell}.mat",
        f"https://raw.githubusercontent.com/Maruthi-prasanth/Battery-Health-Prophecy/master/{cell}.mat",
    ]
    for cell in NASA_CELLS
}

CALCE_ZIP_URLS = {
    cell: f"https://web.calce.umd.edu/batteries/data/{cell}.zip"
    for cell in CALCE_CELLS
}

CALCE_CSV_MIRRORS = {
    cell: f"https://raw.githubusercontent.com/konkon3249/BatteryLifePrediction/master/{cell}.csv"
    for cell in CALCE_CELLS
}


# =====================================================================
# 2. Automated Idempotent Downloader
# =====================================================================

def download_file_with_progress(url: str, dest_path: Path, desc: str = "") -> bool:
    """
    Streams download with progress bar and atomic file write.
    """
    tmp_path = dest_path.with_suffix(".tmp")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        
        with open(tmp_path, "wb") as f, tqdm(
            desc=desc or dest_path.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
                    
        tmp_path.replace(dest_path)
        return True
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False


def ensure_datasets_downloaded(
    nasa_cells: Optional[List[str]] = None,
    calce_cells: Optional[List[str]] = None,
    download_calce_zips: bool = True,
    download_calce_csvs: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Ensures all NASA and CALCE raw files are present locally.
    Idempotent: skips existing verified files.
    """
    nasa_cells = nasa_cells or NASA_CELLS
    calce_cells = calce_cells or CALCE_CELLS
    
    DATA_RAW_NASA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW_CALCE_DIR.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(NASA_CITATION)
        print(CALCE_CITATION)
        print("=" * 70)
        print(" Starting Idempotent Dataset Download & Verification")
        print("=" * 70)

    download_status = {"nasa": {}, "calce": {}}

    # 1. Download NASA .mat files
    print("\n[1/2] Verifying NASA PCoE .mat Files:")
    for cell in nasa_cells:
        dest_mat = DATA_RAW_NASA_DIR / f"{cell}.mat"
        if dest_mat.exists() and dest_mat.stat().st_size > 1024 * 1024:
            print(f"  [OK] {cell}.mat already present ({dest_mat.stat().st_size / (1024*1024):.2f} MB)")
            download_status["nasa"][cell] = True
            continue

        print(f"  * Downloading {cell}.mat...")
        success = False
        for url in NASA_MIRRORS.get(cell, []):
            if download_file_with_progress(url, dest_mat, desc=f"NASA {cell}.mat"):
                success = True
                break
                
        download_status["nasa"][cell] = success
        if success:
            print(f"  [OK] Downloaded {cell}.mat ({dest_mat.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"  [FAILED] Could not acquire {cell}.mat")

    # 2. Download CALCE Files
    print("\n[2/2] Verifying CALCE University of Maryland Datasets:")
    for cell in calce_cells:
        cell_dir = DATA_RAW_CALCE_DIR / cell
        zip_path = DATA_RAW_CALCE_DIR / f"{cell}.zip"
        csv_path = DATA_RAW_CALCE_DIR / f"{cell}.csv"
        
        # 2a. Download summary CSV mirror
        if download_calce_csvs:
            if not (csv_path.exists() and csv_path.stat().st_size > 1024):
                csv_url = CALCE_CSV_MIRRORS.get(cell)
                if csv_url:
                    download_file_with_progress(csv_url, csv_path, desc=f"CALCE {cell}.csv")

        # 2b. Download and unpack official raw Zip archive
        has_xlsx = cell_dir.exists() and len(list(cell_dir.rglob("*.xlsx"))) > 0
        if has_xlsx:
            xlsx_count = len(list(cell_dir.rglob("*.xlsx")))
            print(f"  [OK] {cell} raw data unpacked ({xlsx_count} Excel files in {cell}/)")
            download_status["calce"][cell] = True
            continue

        if download_calce_zips:
            if not (zip_path.exists() and zip_path.stat().st_size > 1024 * 1024):
                zip_url = CALCE_ZIP_URLS.get(cell)
                print(f"  * Downloading official {cell}.zip from CALCE UMD...")
                download_file_with_progress(zip_url, zip_path, desc=f"CALCE {cell}.zip")

            if zip_path.exists() and zip_path.stat().st_size > 1024 * 1024:
                print(f"  * Extracting {cell}.zip into {cell_dir}...")
                cell_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(cell_dir)
                    xlsx_count = len(list(cell_dir.rglob("*.xlsx")))
                    print(f"  [OK] Extracted {cell}.zip ({xlsx_count} Excel files)")
                    download_status["calce"][cell] = True
                except Exception as e:
                    print(f"  [ERROR] Failed to extract {cell}.zip: {e}")
                    download_status["calce"][cell] = False
            else:
                download_status["calce"][cell] = csv_path.exists()
        else:
            download_status["calce"][cell] = csv_path.exists()

    print("\n" + "=" * 70)
    print(" Dataset Synchronization Complete.")
    print("=" * 70)
    return download_status


# =====================================================================
# 3. Raw File Integrity Inspector
# =====================================================================

def inspect_nasa_file(mat_path: Path) -> Dict[str, Any]:
    """
    Inspects a single NASA .mat file:
    Counts struct types and verifies critical time-series columns.
    """
    cell_id = mat_path.stem
    mat = scipy.io.loadmat(str(mat_path))
    if cell_id not in mat:
        raise KeyError(f"Expected top-level key '{cell_id}' in {mat_path.name}")

    cycles = mat[cell_id][0, 0]["cycle"][0]
    total_structs = len(cycles)
    
    charge_count = 0
    discharge_count = 0
    impedance_count = 0
    
    has_v = False
    has_i = False
    has_t = False
    has_time = False
    has_cap = False
    
    initial_cap = None
    final_cap = None
    
    for c in cycles:
        c_type = str(c["type"][0])
        if c_type == "charge":
            charge_count += 1
        elif c_type == "discharge":
            discharge_count += 1
            data_fields = c["data"].dtype.names or ()
            
            if "Voltage_measured" in data_fields:
                has_v = True
            if "Current_measured" in data_fields:
                has_i = True
            if "Temperature_measured" in data_fields:
                has_t = True
            if "Time" in data_fields:
                has_time = True
            if "Capacity" in data_fields:
                has_cap = True
                cap_val = float(c["data"]["Capacity"][0, 0][0, 0])
                if initial_cap is None:
                    initial_cap = cap_val
                final_cap = cap_val
        elif c_type == "impedance":
            impedance_count += 1

    return {
        "cell": cell_id,
        "file_size_mb": mat_path.stat().st_size / (1024 * 1024),
        "total_structs": total_structs,
        "charge_cycles": charge_count,
        "discharge_cycles": discharge_count,
        "impedance_cycles": impedance_count,
        "has_voltage": has_v,
        "has_current": has_i,
        "has_temperature": has_t,
        "has_time": has_time,
        "has_capacity": has_cap,
        "initial_capacity_ah": initial_cap,
        "final_capacity_ah": final_cap,
        "nominal_capacity_ah": NASA_NOMINAL_CAP,
    }


def inspect_calce_cell(cell_id: str) -> Dict[str, Any]:
    """
    Inspects CALCE raw Excel files and summary CSV:
    Verifies Cycle_Index, Step_Index, Current, Voltage, and Capacity columns.
    """
    cell_dir = DATA_RAW_CALCE_DIR / cell_id
    csv_path = DATA_RAW_CALCE_DIR / f"{cell_id}.csv"
    
    xlsx_files = list(cell_dir.rglob("*.xlsx")) if cell_dir.exists() else []
    
    has_cycle_idx = False
    has_step_idx = False
    has_voltage = False
    has_current = False
    has_capacity = False
    
    excel_valid = False
    excel_sheets_checked = 0
    total_xlsx_files = len(xlsx_files)
    
    # Check first few Excel files if present
    if xlsx_files:
        sample_file = xlsx_files[0]
        try:
            xl = pd.ExcelFile(sample_file)
            for sheet_name in xl.sheet_names:
                if sheet_name.lower() == "info":
                    continue
                df_sample = pd.read_excel(sample_file, sheet_name=sheet_name, nrows=10)
                cols = [c.lower() for c in df_sample.columns]
                
                if any("cycle" in c for c in cols):
                    has_cycle_idx = True
                if any("step" in c for c in cols):
                    has_step_idx = True
                if any("volt" in c for c in cols):
                    has_voltage = True
                if any("curr" in c for c in cols):
                    has_current = True
                if any("capac" in c for c in cols):
                    has_capacity = True
                    
                excel_sheets_checked += 1
                excel_valid = True
                break
        except Exception as e:
            excel_valid = False

    # Check CSV mirror
    csv_valid = False
    csv_cycles = 0
    initial_cap = None
    final_cap = None
    
    if csv_path.exists():
        try:
            df_csv = pd.read_csv(csv_path)
            csv_cols = [c.lower() for c in df_csv.columns]
            if "cycle" in csv_cols and "capacity" in csv_cols:
                csv_valid = True
                csv_cycles = len(df_csv)
                initial_cap = float(df_csv["capacity"].iloc[0])
                final_cap = float(df_csv["capacity"].iloc[-1])
                has_cycle_idx = True
                has_capacity = True
        except Exception:
            csv_valid = False

    return {
        "cell": cell_id,
        "total_xlsx_files": total_xlsx_files,
        "excel_valid": excel_valid,
        "csv_valid": csv_valid,
        "csv_cycles": csv_cycles,
        "has_cycle_index": has_cycle_idx,
        "has_step_index": has_step_idx,
        "has_voltage": has_voltage,
        "has_current": has_current,
        "has_capacity": has_capacity,
        "initial_capacity_ah": initial_cap,
        "final_capacity_ah": final_cap,
        "nominal_capacity_ah": CALCE_NOMINAL_CAP,
    }


def inspect_raw_files(verbose: bool = True) -> Dict[str, Any]:
    """
    Performs full data integrity audit across NASA and CALCE datasets.
    Prints formatted verification summary tables.
    """
    nasa_reports = []
    for cell in NASA_CELLS:
        mat_file = DATA_RAW_NASA_DIR / f"{cell}.mat"
        if mat_file.exists():
            nasa_reports.append(inspect_nasa_file(mat_file))
            
    calce_reports = []
    for cell in CALCE_CELLS:
        calce_reports.append(inspect_calce_cell(cell))
        
    if verbose:
        print("\n" + "=" * 90)
        print(" NASA PCoE Battery Dataset Integrity Audit")
        print("=" * 90)
        print(f"{'Cell':<8} {'Size (MB)':<10} {'Structs':<9} {'Charge':<8} {'Discharge':<10} {'Impedance':<10} {'V/I/T/Cap':<12} {'Init Cap (Ah)':<14} {'Final Cap (Ah)':<14}")
        print("-" * 90)
        for r in nasa_reports:
            fields_ok = "All Present" if (r["has_voltage"] and r["has_current"] and r["has_temperature"] and r["has_capacity"]) else "Incomplete"
            print(f"{r['cell']:<8} {r['file_size_mb']:<10.2f} {r['total_structs']:<9} {r['charge_cycles']:<8} {r['discharge_cycles']:<10} {r['impedance_cycles']:<10} {fields_ok:<12} {r['initial_capacity_ah']:<14.4f} {r['final_capacity_ah']:<14.4f}")
        print("-" * 90)

        print("\n" + "=" * 90)
        print(" CALCE Battery Dataset Integrity Audit")
        print("=" * 90)
        print(f"{'Cell':<8} {'Excel Files':<12} {'Step/V/I':<12} {'CSV Valid':<10} {'Total Cycles':<14} {'Init Cap (Ah)':<14} {'Final Cap (Ah)':<14}")
        print("-" * 90)
        for r in calce_reports:
            step_ok = "Verified" if (r["has_step_index"] and r["has_voltage"]) else "CSV Level"
            print(f"{r['cell']:<8} {r['total_xlsx_files']:<12} {step_ok:<12} {str(r['csv_valid']):<10} {r['csv_cycles']:<14} {r['initial_capacity_ah'] or 0.0:<14.4f} {r['final_capacity_ah'] or 0.0:<14.4f}")
        print("=" * 90)

    return {
        "nasa": nasa_reports,
        "calce": calce_reports,
    }


if __name__ == "__main__":
    ensure_datasets_downloaded(verbose=True)
    inspect_raw_files(verbose=True)
