import json
import cv2
import numpy as np

# Input/Output paths
IMAGE_PATH = 'optimization/input/n.png'
OUTPUT_JSON = 'calibration_metrics.json'
OUTPUT_IMAGE = 'calibration_bbox.png'

# Load image in grayscale
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Error: {IMAGE_PATH} not found.")

h, w = img.shape

# 1. Extract a Guaranteed Central 256x256 Patch
P_SIZE = 256
if h < P_SIZE or w < P_SIZE:
    raise ValueError(f"Image dimensions ({w}x{h}) are smaller than required patch size ({P_SIZE}x{P_SIZE}).")

cy_img, cx_img = h // 2, w // 2
y1 = cy_img - P_SIZE // 2
y2 = y1 + P_SIZE
x1 = cx_img - P_SIZE // 2
x2 = x1 + P_SIZE

patch = img[y1:y2, x1:x2]
assert patch.shape == (P_SIZE, P_SIZE), f"Expected ({P_SIZE}, {P_SIZE}), got {patch.shape}"

# Save visualization of selected central area
vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red Bounding Box
cv2.imwrite(OUTPUT_IMAGE, vis_img)

# =====================================================================
# CALIBRATION PASS 1: GABOR & BASELINE STATISTICAL DISTRIBUTIONS
# =====================================================================
# A. Extract primary FFT metrics for Gabor kernel sizing
f_transform = np.fft.rfft2(patch)
magnitude_spectrum = np.abs(f_transform)

magnitude_spectrum[:15, :15] = 0
magnitude_spectrum[-14:, :15] = 0

flat_indices = np.argsort(magnitude_spectrum.flatten())[::-1]
patch_h, patch_w = magnitude_spectrum.shape 

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

# B. Extract Baseline Local Variance Statistics
V_WIN = 15
patch_f = patch.astype(np.float32)
patch_mean = cv2.blur(patch_f, (V_WIN, V_WIN))
patch_mean2 = cv2.blur(patch_f**2, (V_WIN, V_WIN))
patch_var = patch_mean2 - (patch_mean ** 2)

calib_var_mean = float(np.mean(patch_var))
calib_var_std = float(np.std(patch_var))

K_SIGMA = 4.0
var_limit = float(calib_var_mean + (K_SIGMA * calib_var_std))

# C. Dynamic Gabor Parameter Calculation
theta = float(np.arctan2(fy, fx))
lambd = float(w / np.sqrt(fx**2 + fy**2))

ksize = int(1.5 * lambd)
if ksize % 2 == 0:
    ksize += 1

sigma = float(0.35 * lambd)
gamma = 1.0

# D. Extract Baseline Structural Energy & Intensity Statistics
kernel_real_calib = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag_calib = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
kernel_real_calib -= kernel_real_calib.mean()
kernel_imag_calib -= kernel_imag_calib.mean()

patch_blur = cv2.GaussianBlur(patch, (251, 251), 0)
patch_norm = cv2.divide(patch, patch_blur, scale=128)

patch_pad_size = ksize // 2
patch_norm_padded = cv2.copyMakeBorder(
    patch_norm, 
    top=patch_pad_size, 
    bottom=patch_pad_size, 
    left=patch_pad_size, 
    right=patch_pad_size, 
    borderType=cv2.BORDER_REFLECT_101
)

f_real_patch_padded = cv2.filter2D(patch_norm_padded, cv2.CV_64F, kernel_real_calib)
f_imag_patch_padded = cv2.filter2D(patch_norm_padded, cv2.CV_64F, kernel_imag_calib)

f_real_patch = f_real_patch_padded[patch_pad_size:-patch_pad_size, patch_pad_size:-patch_pad_size]
f_imag_patch = f_imag_patch_padded[patch_pad_size:-patch_pad_size, patch_pad_size:-patch_pad_size]

patch_struct_energy = np.sqrt(f_real_patch**2 + f_imag_patch**2)

calib_struct_mean = float(np.mean(patch_struct_energy))
calib_struct_std = float(np.std(patch_struct_energy))

calib_int_mean = float(np.mean(patch_norm))
calib_int_std = float(np.std(patch_norm))

# =====================================================================
# CALIBRATION PASS 2: NWER SPECTRAL PEAK EXTRACTION
# =====================================================================
f_2d = np.fft.fft2(patch.astype(np.float32))
fshift_2d = np.fft.fftshift(f_2d)
mag_sq = np.abs(fshift_2d) ** 2

cy, cx = P_SIZE // 2, P_SIZE // 2

# Zero out DC central area (r <= 8)
y_grid, x_grid = np.ogrid[:P_SIZE, :P_SIZE]
dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
mag_sq[dist_from_center <= 8] = 0

# Extract top 12 spectral peaks corresponding to weave periodicity
TOP_N_PEAKS = 12
flat_indices_nwer = np.argsort(mag_sq.flatten())[::-1][:TOP_N_PEAKS]

peak_coords_normalized = []
PEAK_RADIUS = 5

for idx in flat_indices_nwer:
    py, px = divmod(idx, P_SIZE)
    # Store normalized frequency coordinates [-0.5, 0.5]
    norm_y = float((py - cy) / P_SIZE)
    norm_x = float((px - cx) / P_SIZE)
    peak_coords_normalized.append((norm_y, norm_x))

# =====================================================================
# PACKAGE & SAVE ALL METRICS TO JSON
# =====================================================================
calibration_metrics = {
    "fx": float(fx),
    "fy": float(fy),
    "theta": theta,
    "lambd": lambd,
    "ksize": ksize,
    "sigma": sigma,
    "gamma": gamma,
    "K_SIGMA": K_SIGMA,
    "V_WIN": V_WIN,
    "calib_var_mean": calib_var_mean,
    "calib_var_std": calib_var_std,
    "var_limit": var_limit,
    "calib_struct_mean": calib_struct_mean,
    "calib_struct_std": calib_struct_std,
    "calib_int_mean": calib_int_mean,
    "calib_int_std": calib_int_std,
    "peak_coords_normalized": peak_coords_normalized,
    "peak_radius_px": PEAK_RADIUS
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(calibration_metrics, f, indent=4)

print(f"[SUCCESS] Calibration complete. Saved unified parameters & {TOP_N_PEAKS} NWER peak frequencies to '{OUTPUT_JSON}'.")