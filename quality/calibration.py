# Calibration using live images to suggest thresholds for quality checks

import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from quality.checks import ImageQualityGate

def run_calibration(image_folder, sample_size=50):
    gate = ImageQualityGate()
    stats = []
    
    # Get list of images
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:sample_size]
    
    print(f"Analyzing {len(image_paths)} live images for calibration...")

    for path in tqdm(image_paths):
        img = cv2.imread(path)
        if img is None: continue
        
        # We call analyze but look at the raw scores
        res = gate.analyze(img)
        stats.append(res.scores)

    # Create a DataFrame for analysis
    df = pd.DataFrame(stats)
    
    print("\n--- Calibration Results ---")
    summary = pd.DataFrame({
        'Mean': df.mean(),
        'Std Dev': df.std(),
        'Min (Worst Case)': df.min(),
        'Max (Best Case)': df.max(),
        'Suggested Threshold (Lower Bound)': df.mean() - (1.5 * df.std())
    })
    print(summary)
    return summary

if __name__ == "__main__":
    # Point this to your validation live folder
    val_live_path = "/Users/yeo_menghan/Documents/ml/eKYC-facial/data/celeba-spoof-mini/CelebA_Spoof-mini/val/live" 
    run_calibration(val_live_path)