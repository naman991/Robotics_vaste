## NORMAL IMAGES PASSED
## 4 oil failed due to zoom drift
## lint all passed
## wrinkles/cuts/holes failed completely. 

import os
import glob
import cv2
import numpy as np

# =====================================================================
# CONFIGURATION & DATASET FILTER TOLERANCES
# =====================================================================
INPUT_FOLDER = "theta"     # Directory containing candidate 4K PNG frames
P_SIZE = 256                       # Central patch size for FFT texture analysis

# Reference Baseline Values (Extracted once from primary calibration)
REF_THETA_DEG = 15             # Reference baseline angle (degrees)
REF_LAMBDA = 13.50                 # Reference periodicity wavelength (pixels)

# Strict Quality Gates (Optimal for Layer 1 Generation & Layer 2 Training)
THETA_TOLERANCE_DEG = 5.0          # Max angular drift (+/- degrees)
SCALE_TOLERANCE_PCT = 15.0         # Max scale/zoom drift (+/- %)

# =====================================================================
# SMART TEXTURE ANALYSIS ENGINE
# =====================================================================
def analyze_frame_quality(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0, 0.0, "FAIL: Image Read Error"

    h, w = img.shape
    if h < P_SIZE or w < P_SIZE:
        return False, 0.0, 0.0, f"FAIL: Image smaller than {P_SIZE}x{P_SIZE}"

    # Extract central patch
    cy, cx = h // 2, w // 2
    y1, x1 = cy - P_SIZE // 2, cx - P_SIZE // 2
    patch = img[y1:y1 + P_SIZE, x1:x1 + P_SIZE]

    # Compute 2D Real FFT
    f_transform = np.fft.rfft2(patch)
    magnitude_spectrum = np.abs(f_transform)

    # Mask DC and extreme low-frequency illumination gradients (r <= 15)
    magnitude_spectrum[:15, :15] = 0
    magnitude_spectrum[-14:, :15] = 0

    # Locate dominant spectral peak corresponding to the twill repeat
    patch_w = magnitude_spectrum.shape[1]
    flat_indices = np.argsort(magnitude_spectrum.flatten())[::-1]

    fx, fy = 0, 0
    for idx in flat_indices:
        y_idx, x_idx = divmod(idx, patch_w)
        curr_fx = x_idx
        curr_fy = y_idx if y_idx <= P_SIZE // 2 else y_idx - P_SIZE

        if curr_fx == 0 and curr_fy == 0:
            continue

        fx = curr_fx * (w / P_SIZE)
        fy = curr_fy * (h / P_SIZE)
        break

    if fx == 0 and fy == 0:
        return False, 0.0, 0.0, "FAIL: Frequency Peak Search Failed"

    # Raw line orientation and wavelength
    raw_theta_rad = np.arctan2(fy, fx)
    raw_theta_deg = np.degrees(raw_theta_rad)
    curr_lambda = w / np.sqrt(fx**2 + fy**2)

    # -----------------------------------------------------------------
    # SMART ORIENTATION INVARIANCE ENGINE
    # -----------------------------------------------------------------
    # 1. Project angle into [0, 180) to ignore anti-parallel direction vector flips
    norm_curr = raw_theta_deg % 180.0
    norm_ref = REF_THETA_DEG % 180.0

    # 2. Account for 90-degree orthogonal shifts (transverse weave vector vs longitudinal)
    #    Computes minimal angular distance across 0 deg, 90 deg, and 180 deg symmetry axes
    dev_direct = abs((norm_curr - norm_ref + 90.0) % 180.0 - 90.0)
    dev_ortho  = abs(((norm_curr + 90.0) % 180.0 - norm_ref + 90.0) % 180.0 - 90.0)

    delta_theta = min(dev_direct, dev_ortho)

    # -----------------------------------------------------------------
    # SCALE / ZOOM SHIFT EVALUATION
    # -----------------------------------------------------------------
    delta_scale_pct = abs((curr_lambda - REF_LAMBDA) / REF_LAMBDA) * 100.0

    # -----------------------------------------------------------------
    # EVALUATE QUALITY GATES
    # -----------------------------------------------------------------
    theta_pass = delta_theta <= THETA_TOLERANCE_DEG
    scale_pass = delta_scale_pct <= SCALE_TOLERANCE_PCT

    if theta_pass and scale_pass:
        return True, norm_curr, curr_lambda, "PASS"
    else:
        reasons = []
        if not theta_pass:
            reasons.append(f"Angle Drift ({delta_theta:.2f}° > {THETA_TOLERANCE_DEG}°)")
        if not scale_pass:
            reasons.append(f"Zoom Drift ({delta_scale_pct:.1f}% > {SCALE_TOLERANCE_PCT}%)")
        return False, norm_curr, curr_lambda, f"FAIL ({', '.join(reasons)})"

# =====================================================================
# MAIN EXECUTOR
# =====================================================================
def main():
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.png")))
    if not image_paths:
        print(f"No PNG images found in '{INPUT_FOLDER}'.")
        return

    print(f"{'Filename':<35} | {'Angle (Deg)':<12} | {'Lambda (px)':<12} | {'Status'}")
    print("-" * 85)

    passed_count = 0
    for path in image_paths:
        filename = os.path.basename(path)
        is_valid, angle, lambd_val, status = analyze_frame_quality(path)

        if is_valid:
            passed_count += 1

        print(f"{filename:<35} | {angle:<12.2f} | {lambd_val:<12.2f} | {status}")

    print("-" * 85)
    print(f"Summary: {passed_count}/{len(image_paths)} frames passed quality gate.")

if __name__ == "__main__":
    main()