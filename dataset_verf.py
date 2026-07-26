#MIN MAX SCALING LOGIC REPLACED WITH Z-SCORE LOGIC FOR STABILITY
# edges are getting flagged for normal images. specifically the left and right edges. 

import cv2
import numpy as np

# 1. Load image in grayscale
# IMAGE_PATH = 'asset/normal/img/n5.png'
IMAGE_PATH = 'asset/wrinkles/w2.png'
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: {IMAGE_PATH} not found.")
    exit()

h, w = img.shape

# 1. Extract a Small Central 256x256 Patch for Lean Calibration
P_SIZE = 256
cx_img, cy_img = w // 2, h // 2
patch = img[cy_img - P_SIZE//2 : cy_img + P_SIZE//2, cx_img - P_SIZE//2 : cx_img + P_SIZE//2]

# =====================================================================
# CALIBRATION PASS: FFT PARAMETERS & BASELINE STATISTICAL DISTRIBUTIONS
# =====================================================================
# A. Extract FFT metrics
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

# B. Extract Baseline Local Variance Statistics from Clean Calibration Patch
V_WIN = 15
patch_f = patch.astype(np.float32)
patch_mean = cv2.blur(patch_f, (V_WIN, V_WIN))
patch_mean2 = cv2.blur(patch_f**2, (V_WIN, V_WIN))
patch_var = patch_mean2 - (patch_mean ** 2)

calib_var_mean = np.mean(patch_var)
calib_var_std = np.std(patch_var)

# K_SIGMA: Dynamic distance threshold above background noise
K_SIGMA = 4.0
var_limit = calib_var_mean + (K_SIGMA * calib_var_std)

# 3. Dynamic Gabor Parameter Calculation
theta = np.arctan2(fy, fx)
lambd = w / np.sqrt(fx**2 + fy**2)

ksize = int(1.5 * lambd)
if ksize % 2 == 0:
    ksize += 1

sigma = 0.35 * lambd
gamma = 1.0

# Extract Baseline Structural Energy & Intensity Statistics from Patch
kernel_real_calib = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag_calib = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
kernel_real_calib -= kernel_real_calib.mean()
kernel_imag_calib -= kernel_imag_calib.mean()

patch_blur = cv2.GaussianBlur(patch, (251, 251), 0)
patch_norm = cv2.divide(patch, patch_blur, scale=128)

f_real_patch = cv2.filter2D(patch_norm, cv2.CV_64F, kernel_real_calib)
f_imag_patch = cv2.filter2D(patch_norm, cv2.CV_64F, kernel_imag_calib)
patch_struct_energy = np.sqrt(f_real_patch**2 + f_imag_patch**2)

calib_struct_mean = np.mean(patch_struct_energy)
calib_struct_std = np.std(patch_struct_energy)

calib_int_mean = np.mean(patch_norm)
calib_int_std = np.std(patch_norm)

# 4. Stream 1 Illumination Normalization Framework
blur = cv2.GaussianBlur(img, (251, 251), 0)

mean_light, std_light = cv2.meanStdDev(blur)
light_uniformity_drift = (std_light[0][0] / mean_light[0][0]) * 100

print(f"Lighting Uniformity Drift: {light_uniformity_drift:.2f}%")

normalized_img = cv2.divide(img, blur, scale=128)

# =====================================================================
# PRODUCTION-GRADE REFLECTIVE PADDING INTEGRATION
# =====================================================================
pad_size = ksize // 2

padded_img = cv2.copyMakeBorder(
    normalized_img, 
    top=pad_size, 
    bottom=pad_size, 
    left=pad_size, 
    right=pad_size, 
    borderType=cv2.BORDER_REFLECT_101
)

# 5. Full-Resolution Structural Energy Map Computation
kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
kernel_real -= kernel_real.mean()
kernel_imag -= kernel_imag.mean()

f_real_padded = cv2.filter2D(padded_img, cv2.CV_64F, kernel_real)
f_imag_padded = cv2.filter2D(padded_img, cv2.CV_64F, kernel_imag)

f_real = f_real_padded[pad_size:-pad_size, pad_size:-pad_size]
f_imag = f_imag_padded[pad_size:-pad_size, pad_size:-pad_size]

structural_energy = np.sqrt(f_real**2 + f_imag**2)

# =====================================================================
# STREAM 1: Z-SCORE LOGIC ENVELOPE (REPLACES UNSTABLE MIN-MAX SCALING)
# =====================================================================
z_struct_energy = (structural_energy - calib_struct_mean) / (calib_struct_std + 1e-7)
struct_fray_mask = z_struct_energy > K_SIGMA
struct_void_mask = z_struct_energy < -K_SIGMA
struct_mask = struct_fray_mask | struct_void_mask

z_intensity = (normalized_img.astype(np.float32) - calib_int_mean) / (calib_int_std + 1e-7)
dark_mask = z_intensity < -K_SIGMA
light_saturation_mask = z_intensity > K_SIGMA
intensity_mask = dark_mask | light_saturation_mask

combined_pixel_mask = (struct_mask | intensity_mask).astype(np.uint8) * 255

CLOSE_KSIZE = 21
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KSIZE, CLOSE_KSIZE))
closed_mask = cv2.morphologyEx(combined_pixel_mask, cv2.MORPH_CLOSE, close_kernel)

LINE_LEN = 15
base_v_kernel = np.zeros((LINE_LEN, LINE_LEN), dtype=np.uint8)
base_v_kernel[:, LINE_LEN // 2] = 1 

theta_deg = np.degrees(theta) + 90.0
M_rot = cv2.getRotationMatrix2D((LINE_LEN / 2.0, LINE_LEN / 2.0), theta_deg, 1.0)
aligned_vert_kernel = cv2.warpAffine(base_v_kernel, M_rot, (LINE_LEN, LINE_LEN), flags=cv2.INTER_NEAREST)

cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, aligned_vert_kernel)

# =====================================================================
# BRANCH 3: LOCAL VARIANCE STREAM (CALIBRATION-RELATIVE THRESHOLD)
# =====================================================================
img_f = img.astype(np.float32)

mean_I = cv2.blur(img_f, (V_WIN, V_WIN))
mean_I2 = cv2.blur(img_f**2, (V_WIN, V_WIN))

local_var = mean_I2 - (mean_I ** 2)

_, raw_oil_mask_float = cv2.threshold(local_var, var_limit, 255, cv2.THRESH_BINARY)
raw_oil_mask = raw_oil_mask_float.astype(np.uint8)

oil_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed_oil_mask = cv2.morphologyEx(raw_oil_mask, cv2.MORPH_CLOSE, oil_close_kernel)

oil_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
solid_oil_mask = cv2.morphologyEx(closed_oil_mask, cv2.MORPH_OPEN, oil_open_kernel)

unified_candidate_mask = cv2.bitwise_or(cleaned_mask, solid_oil_mask)
# =====================================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(unified_candidate_mask)

MIN_CLUSTER_SIZE = 120
final_macro_mask = np.zeros_like(unified_candidate_mask)
defect_count = 0
max_size = 0

for i in range(1, num_labels):
    cluster_size = stats[i, cv2.CC_STAT_AREA]
    if cluster_size >= MIN_CLUSTER_SIZE:
        final_macro_mask[labels == i] = 255
        defect_count += 1
        if cluster_size > max_size:
            max_size = cluster_size

# =====================================================================
# DIAGNOSTIC OUTPUT DUMPS
# =====================================================================
cv2.imwrite('diag_01_gauss_blur.png', blur)
cv2.imwrite('diag_01.5_combined.png', combined_pixel_mask)
cv2.imwrite('diag_02_morph_close.png', closed_mask)
cv2.imwrite('diag_03_morph_open.png', cleaned_mask)

norm_var_vis = np.clip((local_var / (var_limit * 1.5)) * 255, 0, 255).astype(np.uint8)
cv2.imwrite('diag_04_variance_map.png', norm_var_vis)

cv2.imwrite('diag_05_oil_solid_filtered.png', solid_oil_mask)
cv2.imwrite('diag_06_unified_candidates.png', unified_candidate_mask)

status = "FAIL" if defect_count > 0 else "PASS"
cv2.imwrite('defect.png', final_macro_mask)

print("=" * 45)
print(" STREAM 1 SHARP INSPECTION REPORT ")
print("=" * 45)
print(f"Calibrated Theta        : {theta:.4f} rad")
print(f"Calibrated Lambda       : {lambd:.4f} px (ksize={ksize})")
print(f"Calibrated Var Base     : Mean={calib_var_mean:.2f}, Std={calib_var_std:.2f}")
print(f"Calibrated Var Limit    : {var_limit:.2f} (K={K_SIGMA})")
print(f"Calibrated Struct Base  : Mean={calib_struct_mean:.2f}, Std={calib_struct_std:.2f}")
print("-" * 45)
print(f"Structural Outliers     : {np.sum(struct_mask)} px")
print(f"Intensity/Stain Outliers: {np.sum(intensity_mask)} px")
print(f"Oil Branch Outliers     : {np.sum(solid_oil_mask > 0)} px")
print("-" * 45)
print(f"Continuous Macro Blocks : {defect_count}")
print(f"Largest Defect Block    : {max_size} px")
print(f"FINAL INSPECTION STATUS : {status}")
print("=" * 45)