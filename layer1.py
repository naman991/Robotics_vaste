import os
import glob
import json
import cv2
import numpy as np

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
INPUT_FOLDER = 'processed_fabric_frames'       # Folder containing input PNG images
OUTPUT_FOLDER = 'output_defects'   # Folder to save output defect masks
CALIB_JSON_PATH = 'calibration_metrics.json'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. Load Pre-Calculated Calibration Metrics
if not os.path.exists(CALIB_JSON_PATH):
    raise FileNotFoundError(f"Error: Calibration file '{CALIB_JSON_PATH}' not found. Run calibration first.")

with open(CALIB_JSON_PATH, 'r') as f:
    calib = json.load(f)

# Extract metrics from JSON
theta = calib["theta"]
lambd = calib["lambd"]
ksize = calib["ksize"]
sigma = calib["sigma"]
gamma = calib["gamma"]
K_SIGMA = calib["K_SIGMA"]
V_WIN = calib["V_WIN"]

var_limit = calib["var_limit"]
calib_var_mean = calib["calib_var_mean"]
calib_var_std = calib["calib_var_std"]

calib_struct_mean = calib["calib_struct_mean"]
calib_struct_std = calib["calib_struct_std"]

calib_int_mean = calib["calib_int_mean"]
calib_int_std = calib["calib_int_std"]

# Pre-build Gabor Kernels
kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
kernel_real -= kernel_real.mean()
kernel_imag -= kernel_imag.mean()

# Pre-build Morphological Kernels
CLOSE_KSIZE = 21
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KSIZE, CLOSE_KSIZE))

LINE_LEN = 15
base_v_kernel = np.zeros((LINE_LEN, LINE_LEN), dtype=np.uint8)
base_v_kernel[:, LINE_LEN // 2] = 1 
theta_deg = np.degrees(theta) + 90.0
M_rot = cv2.getRotationMatrix2D((LINE_LEN / 2.0, LINE_LEN / 2.0), theta_deg, 1.0)
aligned_vert_kernel = cv2.warpAffine(base_v_kernel, M_rot, (LINE_LEN, LINE_LEN), flags=cv2.INTER_NEAREST)

oil_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
oil_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# =====================================================================
# BATCH PROCESSING
# =====================================================================
image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.png'))

if not image_paths:
    print(f"No PNG files found in '{INPUT_FOLDER}'.")
    exit()

print(f"Processing {len(image_paths)} images from '{INPUT_FOLDER}'...\n")

for img_path in image_paths:
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Skipping corrupt image: {img_path}")
        continue

    h, w = img.shape

    # 1. Stream 1 Illumination Normalization Framework
    blur_pad = 125
    padded_raw_img = cv2.copyMakeBorder(img, blur_pad, blur_pad, blur_pad, blur_pad, borderType=cv2.BORDER_REFLECT_101)
    blur_padded = cv2.GaussianBlur(padded_raw_img, (251, 251), 0)
    blur = blur_padded[blur_pad:-blur_pad, blur_pad:-blur_pad]

    normalized_img = cv2.divide(img, blur, scale=128)

    # 2. Production-Grade Reflective Padding Integration
    pad_size = ksize
    padded_img = cv2.copyMakeBorder(
        normalized_img, 
        top=pad_size, 
        bottom=pad_size, 
        left=pad_size, 
        right=pad_size, 
        borderType=cv2.BORDER_REFLECT_101
    )

    # 3. Full-Resolution Structural Energy Map Computation
    f_real_padded = cv2.filter2D(padded_img, cv2.CV_64F, kernel_real)
    f_imag_padded = cv2.filter2D(padded_img, cv2.CV_64F, kernel_imag)

    f_real = f_real_padded[pad_size:-pad_size, pad_size:-pad_size]
    f_imag = f_imag_padded[pad_size:-pad_size, pad_size:-pad_size]

    structural_energy = np.sqrt(f_real**2 + f_imag**2)

    # 4. Stream 1: Z-Score Logic Envelope
    col_energy_mean = cv2.GaussianBlur(np.mean(structural_energy, axis=0, keepdims=True), (101, 1), 0)
    col_int_mean = cv2.GaussianBlur(np.mean(normalized_img.astype(np.float32), axis=0, keepdims=True), (101, 1), 0)

    z_struct_energy = (structural_energy - col_energy_mean) / (calib_struct_std + 1e-7)
    struct_fray_mask = z_struct_energy > K_SIGMA
    struct_void_mask = z_struct_energy < -K_SIGMA
    struct_mask = struct_fray_mask | struct_void_mask

    z_intensity = (normalized_img.astype(np.float32) - col_int_mean) / (calib_int_std + 1e-7)
    dark_mask = z_intensity < -K_SIGMA
    light_saturation_mask = z_intensity > K_SIGMA
    intensity_mask = dark_mask | light_saturation_mask

    combined_pixel_mask = (struct_mask | intensity_mask).astype(np.uint8) * 255

    closed_mask = cv2.morphologyEx(combined_pixel_mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, aligned_vert_kernel)

    # 5. Branch 3: Local Variance Stream
    img_f = img.astype(np.float32)
    mean_I = cv2.blur(img_f, (V_WIN, V_WIN))
    mean_I2 = cv2.blur(img_f**2, (V_WIN, V_WIN))
    local_var = mean_I2 - (mean_I ** 2)

    _, raw_oil_mask_float = cv2.threshold(local_var, var_limit, 255, cv2.THRESH_BINARY)
    raw_oil_mask = raw_oil_mask_float.astype(np.uint8)

    closed_oil_mask = cv2.morphologyEx(raw_oil_mask, cv2.MORPH_CLOSE, oil_close_kernel)
    solid_oil_mask = cv2.morphologyEx(closed_oil_mask, cv2.MORPH_OPEN, oil_open_kernel)

    unified_candidate_mask = cv2.bitwise_or(cleaned_mask, solid_oil_mask)

    # 6. Connected Component Size Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unified_candidate_mask)

    MIN_CLUSTER_SIZE = 270
    final_macro_mask = np.zeros_like(unified_candidate_mask)
    defect_count = 0

    for i in range(1, num_labels):
        cluster_size = stats[i, cv2.CC_STAT_AREA]
        if cluster_size >= MIN_CLUSTER_SIZE:
            final_macro_mask[labels == i] = 255
            defect_count += 1

    # Save output using original image's filename in the output folder
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, final_macro_mask)

    status = "FAIL" if defect_count > 0 else "PASS"
    print(f"[{status}] {filename} -> Saved defect mask to {output_path}")

print("\nBatch processing complete.")