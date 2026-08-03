import os
import glob
import json
import cv2
import numpy as np
from collections import defaultdict

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
INPUT_FOLDER = 'input'        # Folder containing input PNG images
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

# Pre-build Spatial Gabor Kernels
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
# BATCH PROCESSING WITH BENCHMARKING
# =====================================================================
image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.png'))

if not image_paths:
    print(f"No PNG files found in '{INPUT_FOLDER}'.")
    exit()

print(f"Processing {len(image_paths)} images from '{INPUT_FOLDER}'...\n")

# Benchmark accumulator (milliseconds)
stage_latencies = defaultdict(list)
freq = cv2.getTickFrequency()

# Cache FFT frequency kernels (lazy initialization on first frame)
dft_kernel_real = None
dft_kernel_imag = None
cached_opt_h, cached_opt_w = None, None

for img_path in image_paths:
    filename = os.path.basename(img_path)
    
    # Stage 0: I/O Read
    t0 = cv2.getTickCount()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    t_io_read = (cv2.getTickCount() - t0) / freq * 1000.0
    
    if img is None:
        print(f"Skipping corrupt image: {img_path}")
        continue

    h, w = img.shape

    # Stage 1: Illumination Normalization (Optimized via 8x Downsampled Background Estimation)
    t0 = cv2.getTickCount()
    ds_factor = 8
    ds_w = w // ds_factor
    ds_h = h // ds_factor

    # Downsample image for background estimation
    img_ds = cv2.resize(img, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

    # Scale padding and kernel size proportionally for downsampled domain
    blur_pad_ds = max(1, 125 // ds_factor)
    ksize_ds = 31  # Scaled down from 251x251 to preserve effective spatial radius

    padded_raw_img_ds = cv2.copyMakeBorder(
        img_ds, blur_pad_ds, blur_pad_ds, blur_pad_ds, blur_pad_ds, borderType=cv2.BORDER_REFLECT_101
    )
    blur_padded_ds = cv2.GaussianBlur(padded_raw_img_ds, (ksize_ds, ksize_ds), 0)
    blur_ds = blur_padded_ds[blur_pad_ds:-blur_pad_ds, blur_pad_ds:-blur_pad_ds]

    # Upsample smooth illumination background map back to original full resolution
    blur = cv2.resize(blur_ds, (w, h), interpolation=cv2.INTER_LINEAR)

    normalized_img = cv2.divide(img, blur, scale=128)
    t_illum_norm = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 2: 2D FFT Frequency-Domain Gabor Filtering
    t0 = cv2.getTickCount()
    pad_size = ksize
    
    # 1. Standard Reflective Padding to avoid boundary reflection issues
    padded_img = cv2.copyMakeBorder(
        normalized_img, 
        top=pad_size, 
        bottom=pad_size, 
        left=pad_size, 
        right=pad_size, 
        borderType=cv2.BORDER_REFLECT_101
    )
    padded_h, padded_w = padded_img.shape

    # 2. Optimal DFT Size Calculation (Target: 2250 x 3920 for 2160x3840 raw image)
    opt_h = cv2.getOptimalDFTSize(padded_h)
    opt_w = cv2.getOptimalDFTSize(padded_w)

    # 3. Pre-compute and cache frequency-domain Gabor kernels once for opt_h x opt_w
    if dft_kernel_real is None or (cached_opt_h, cached_opt_w) != (opt_h, opt_w):
        # Zero-pad spatial Gabor kernels to optimal DFT size
        kernel_real_padded = np.zeros((opt_h, opt_w), dtype=np.float64)
        kernel_imag_padded = np.zeros((opt_h, opt_w), dtype=np.float64)

        kernel_real_padded[:ksize, :ksize] = kernel_real
        kernel_imag_padded[:ksize, :ksize] = kernel_imag

        # Compute complex DFTs of pre-padded kernels
        dft_kernel_real = cv2.dft(kernel_real_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_kernel_imag = cv2.dft(kernel_imag_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        cached_opt_h, cached_opt_w = opt_h, opt_w

    # 4. Zero-pad reflective padded image to optimal DFT size
    padded_img_opt = cv2.copyMakeBorder(
        padded_img, 
        top=0, 
        bottom=opt_h - padded_h, 
        left=0, 
        right=opt_w - padded_w, 
        borderType=cv2.BORDER_CONSTANT, 
        value=0
    )

    # 5. Forward 2D FFT on input image
    dft_img = cv2.dft(padded_img_opt.astype(np.float64), flags=cv2.DFT_COMPLEX_OUTPUT)

    # 6. Frequency domain complex multiplication (Convolution Theorem)
    dft_res_real = cv2.mulSpectrums(dft_img, dft_kernel_real, flags=0)
    dft_res_imag = cv2.mulSpectrums(dft_img, dft_kernel_imag, flags=0)

    # 7. Inverse 2D FFT back to spatial space
    idft_real = cv2.idft(dft_res_real, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    idft_imag = cv2.idft(dft_res_imag, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # 8. Unpad optimal DFT canvas -> Unpad reflective border -> Exact (h, w) match
    f_real = idft_real[:padded_h, :padded_w][pad_size:-pad_size, pad_size:-pad_size]
    f_imag = idft_imag[:padded_h, :padded_w][pad_size:-pad_size, pad_size:-pad_size]

    structural_energy = np.sqrt(f_real**2 + f_imag**2)
    t_gabor = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 3: Z-Score Envelope & Thresholding
    t0 = cv2.getTickCount()
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
    t_zscore = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 4: Structural Morphological Cleaning
    t0 = cv2.getTickCount()
    closed_mask = cv2.morphologyEx(combined_pixel_mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, aligned_vert_kernel)
    t_struct_morph = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 5: Local Variance Stream
    t0 = cv2.getTickCount()
    img_f = img.astype(np.float32)
    mean_I = cv2.blur(img_f, (V_WIN, V_WIN))
    mean_I2 = cv2.blur(img_f**2, (V_WIN, V_WIN))
    local_var = mean_I2 - (mean_I ** 2)

    _, raw_oil_mask_float = cv2.threshold(local_var, var_limit, 255, cv2.THRESH_BINARY)
    raw_oil_mask = raw_oil_mask_float.astype(np.uint8)

    closed_oil_mask = cv2.morphologyEx(raw_oil_mask, cv2.MORPH_CLOSE, oil_close_kernel)
    solid_oil_mask = cv2.morphologyEx(closed_oil_mask, cv2.MORPH_OPEN, oil_open_kernel)

    unified_candidate_mask = cv2.bitwise_or(cleaned_mask, solid_oil_mask)
    t_variance_stream = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 6: Connected Components Size Filtering
    t0 = cv2.getTickCount()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unified_candidate_mask)

    MIN_CLUSTER_SIZE = 120
    final_macro_mask = np.zeros_like(unified_candidate_mask)
    defect_count = 0

    for i in range(1, num_labels):
        cluster_size = stats[i, cv2.CC_STAT_AREA]
        if cluster_size >= MIN_CLUSTER_SIZE:
            final_macro_mask[labels == i] = 255
            defect_count += 1
    t_cc_filter = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 7: I/O Write
    t0 = cv2.getTickCount()
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, final_macro_mask)
    t_io_write = (cv2.getTickCount() - t0) / freq * 1000.0

    # Accumulate metrics
    stage_latencies["01_io_read"].append(t_io_read)
    stage_latencies["02_illum_norm"].append(t_illum_norm)
    stage_latencies["03_gabor_filter"].append(t_gabor)
    stage_latencies["04_zscore_envelope"].append(t_zscore)
    stage_latencies["05_struct_morph"].append(t_struct_morph)
    stage_latencies["06_variance_stream"].append(t_variance_stream)
    stage_latencies["07_cc_filtering"].append(t_cc_filter)
    stage_latencies["08_io_write"].append(t_io_write)

    total_image_latency = (
        t_io_read + t_illum_norm + t_gabor + t_zscore + 
        t_struct_morph + t_variance_stream + t_cc_filter + t_io_write
    )
    stage_latencies["00_total"].append(total_image_latency)

    status = "FAIL" if defect_count > 0 else "PASS"
    print(f"[{status}] {filename} -> Latency: {total_image_latency:.2f} ms")

# =====================================================================
# BENCHMARK REPORT SUMMARY
# =====================================================================
num_images = len(stage_latencies["00_total"])
avg_total = np.mean(stage_latencies["00_total"])

print("\n" + "=" * 65)
print(f" LATENCY BENCHMARK REPORT ({num_images} Images Processed)")
print("=" * 65)
print(f"{'Pipeline Stage':<28} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Share (%)':<8}")
print("-" * 65)

for key in sorted(stage_latencies.keys()):
    if key == "00_total":
        continue
    times = stage_latencies[key]
    avg_t = np.mean(times)
    min_t = np.min(times)
    max_t = np.max(times)
    pct = (avg_t / avg_total) * 100.0
    
    stage_name = key[3:].replace('_', ' ').title()
    print(f"{stage_name:<28} | {avg_t:<10.2f} | {min_t:<10.2f} | {max_t:<10.2f} | {pct:<8.1f}")

print("-" * 65)
print(f"{'TOTAL PIPELINE (per frame)':<28} | {avg_total:<10.2f} | {np.min(stage_latencies['00_total']):<10.2f} | {np.max(stage_latencies['00_total']):<10.2f} | 100.0%")
print("=" * 65)