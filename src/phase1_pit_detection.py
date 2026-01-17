"""
Phase 1: Pit Detection - Enhanced with CLAHE for Low-Contrast Soil

Updates:
1. Added CLAHE (Contrast Boosting) to make pits visible in bright soil.
2. Tuned Vegetation Mask to be less aggressive (so it doesn't mask brown soil).
3. Adjusted Thresholding to capture fainter shadows.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import rasterio

# --- CONFIGURATION ---
PIT_SIDE_M = 0.45
MIN_ASPECT = 0.4        # Looser aspect ratio
MAX_ASPECT = 2.5
MIN_SOLIDITY = 0.55     # Allow messier shapes
THRESHOLD_BLOCK = 101   # LARGER block size (101) handles large soil patches better
THRESHOLD_C = 5         # Tuned for CLAHE output

def get_vegetation_mask(r, g, b, threshold=20):
    """
    Creates a boolean mask where vegetation exists.
    Increased threshold to 20 to avoid masking brownish/greenish soil.
    """
    r_int = r.astype(np.int16)
    g_int = g.astype(np.int16)
    b_int = b.astype(np.int16)
    exg = (2 * g_int) - r_int - b_int
    return exg > threshold

def detect_pits_from_file(tiff_path, site_name="Unknown", min_solidity=MIN_SOLIDITY):
    tiff_path = Path(tiff_path)
    if not tiff_path.exists():
        print(f"Error: File not found at {tiff_path}")
        return None

    print(f"Processing: {tiff_path.name}")

    with rasterio.open(tiff_path) as src:
        res_m = src.res[0]
        expected_width_px = PIT_SIDE_M / res_m
        target_area_px = expected_width_px ** 2
        
        # Keep area filter loose
        min_area_px = target_area_px * 0.15
        max_area_px = target_area_px * 3.0
        
        print(f"  Resolution: {res_m*100:.2f} cm/px")
        print(f"  Area Filter: {min_area_px:.0f} - {max_area_px:.0f} px²")

        # 1. Band Selection
        if src.count >= 3:
            r = src.read(1)
            g = src.read(2)
            b = src.read(3)
            detection_band = r # Red is best for soil
            veg_mask = get_vegetation_mask(r, g, b, threshold=20) # Weaker mask
        else:
            detection_band = src.read(1)
            veg_mask = None

        # 2. CLAHE PRE-PROCESSING (The Fix)
        # This boosts contrast in the bright central areas
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        # Normalize to 0-255 first
        img_norm = cv2.normalize(detection_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE
        img_boosted = clahe.apply(img_norm)
        
        # Mask vegetation (Make it white = ignored)
        if veg_mask is not None:
            img_boosted[veg_mask] = 255

        # 3. Adaptive Thresholding
        # Invert: Pits (Dark) -> Bright
        img_inv = cv2.bitwise_not(img_boosted)
        
        binary_map = cv2.adaptiveThreshold(
            img_inv,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            THRESHOLD_BLOCK,
            THRESHOLD_C,
        )

        # 4. Cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
        
        # 5. Contours
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_pits = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area_px < area < max_area_px): continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w)/h
            if not (MIN_ASPECT < aspect < MAX_ASPECT): continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = area / hull_area
            if solidity < min_solidity: continue

            # Confirmed Pit
            cx = x + w // 2
            cy = y + h // 2
            lon, lat = src.xy(cy, cx)

            detected_pits.append({
                "site": site_name,
                "pit_id": len(detected_pits) + 1,
                "lat": lat,
                "lon": lon,
                "pixel_x": cx,
                "pixel_y": cy,
                "area_px": area,
                "solidity": round(solidity, 2)
            })

    df = pd.DataFrame(detected_pits)
    print(f"  -> Detected {len(df)} confirmed pits.")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to orthomosaic .tif")
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    site_name = input_path.parent.name if input_path.parent.name != "input" else input_path.stem
    
    print(f"Site: {site_name}")
    df = detect_pits_from_file(str(input_path), site_name=site_name)
    
    if df is not None and not df.empty:
        output_dir = Path("output") / site_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        out_csv = output_dir / "map.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV to {out_csv}")
        
        # --- NEW: Generate Debug Visual Overlay Immediately ---
        # This saves an image with dots so you don't have to wait for QGIS
        import matplotlib.pyplot as plt
        
        # Load image again for plotting
        with rasterio.open(str(input_path)) as src:
            if src.count >= 3:
                img_vis = np.dstack((src.read(1), src.read(2), src.read(3)))
            else:
                img_vis = src.read(1)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(img_vis)
        plt.scatter(df['pixel_x'], df['pixel_y'], c='red', s=10, alpha=0.6)
        plt.axis('off')
        plt.title(f"Detected {len(df)} Pits")
        
        out_png = output_dir / "debug_overlay.png"
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Saved Debug Image to {out_png}")
        
    else:
        print("No pits detected. Try lowering MIN_SOLIDITY or adjusting THRESHOLD_C.")