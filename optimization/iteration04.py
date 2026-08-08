# # final cheap python based optimization, latency reported: 3935ms
#   - eliminated python loop that runs N (number of white connected blobs) times to paint the final binary mask. 
#    / instead now using 1D Label remapping using cv2.LUT transform, where one single SIMD instruction pass gives binary mask. 
#   - column averages for fft energy when comparing for each column is done on a downscaled(8x) image, as intensity remains unaffected(drift by 0.15%) 
#   - downsampled 2D FFT Gabor filtering is done on 1080p image, instead of 4K, and upscaled back to 4k. satisfies nyquist limit as lambda is more than 4; and min defect size is 270px. 
#    / resultant latency: 2142s( gabor filter dropped 2.4s --> 0.5s) 


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

# =====================================================================
# STAGE 2 DOWNSAMPLING PARAMETER SCALING (2x Spatial Reduction)
# =====================================================================
DS_STAGE2 = 2

# Scale wavelength and sigma down by 2x
lambd_ds = lambd / DS_STAGE2
sigma_ds = sigma / DS_STAGE2

# Scale ksize down by 2x and force odd integer
ksize_ds = int(ksize / DS_STAGE2)
if ksize_ds % 2 == 0:
    ksize_ds += 1
ksize_ds = max(3, ksize_ds)

# Pre-build Downsampled Spatial Gabor Kernels (float32 precision)
kernel_real = cv2.getGaborKernel((ksize_ds, ksize_ds), sigma_ds, theta, lambd_ds, gamma, psi=0, ktype=cv2.CV_32F)
kernel_imag = cv2.getGaborKernel((ksize_ds, ksize_ds), sigma_ds, theta, lambd_ds, gamma, psi=np.pi/2, ktype=cv2.CV_32F)
kernel_real -= kernel_real.mean()
kernel_imag -= kernel_imag.mean()

# Pre-build Morphological Kernels
CLOSE_KSIZE = 21
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KSIZE, CLOSE_KSIZE))

LINE_LEN = 15
if LINE_LEN % 2 == 0:
    LINE_LEN += 1

center_idx = LINE_LEN // 2
base_v_kernel = np.zeros((LINE_LEN, LINE_LEN), dtype=np.uint8)
base_v_kernel[:, center_idx] = 1 

theta_deg = np.degrees(theta) + 90.0
rot_center = (float(center_idx), float(center_idx))
M_rot = cv2.getRotationMatrix2D(rot_center, theta_deg, 1.0)

aligned_vert_kernel = cv2.warpAffine(
    base_v_kernel, 
    M_rot, 
    (LINE_LEN, LINE_LEN), 
    flags=cv2.INTER_NEAREST,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)
aligned_vert_kernel = (aligned_vert_kernel > 0).astype(np.uint8)

oil_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
oil_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# =====================================================================
# GLOBAL DFT INITIALIZATION & STATIC BUFFER PRE-ALLOCATION (SYSTEM BOOT)
# =====================================================================
# Target downsampled dimensions: 1080x1920 -> padded with ksize_ds -> optimal canvas
pad_size_ds = ksize_ds
raw_padded_h_ds = (2160 // DS_STAGE2) + (2 * pad_size_ds)
raw_padded_w_ds = (3840 // DS_STAGE2) + (2 * pad_size_ds)

opt_h = cv2.getOptimalDFTSize(raw_padded_h_ds)
opt_w = cv2.getOptimalDFTSize(raw_padded_w_ds)

# Pre-compute CCS real frequency domain Gabor kernels ONCE at boot
kernel_real_padded = np.zeros((opt_h, opt_w), dtype=np.float32)
kernel_imag_padded = np.zeros((opt_h, opt_w), dtype=np.float32)

kernel_real_padded[:ksize_ds, :ksize_ds] = kernel_real
kernel_imag_padded[:ksize_ds, :ksize_ds] = kernel_imag

dft_kernel_real = cv2.dft(kernel_real_padded, flags=cv2.DFT_REAL_OUTPUT)
dft_kernel_imag = cv2.dft(kernel_imag_padded, flags=cv2.DFT_REAL_OUTPUT)

# PRE-COMPUTE SCALAR RECIPROCALS: Replace per-frame array divisions with scalar multiplications
inv_struct_std = np.float32(1.0 / (calib_struct_std + 1e-7))
inv_int_std = np.float32(1.0 / (calib_int_std + 1e-7))

# STATIC BUFFERS: Eliminate dynamic allocations on RAM bus during runtime
padded_img_opt_buf = np.zeros((opt_h, opt_w), dtype=np.float32)
dft_img_buf = np.zeros((opt_h, opt_w), dtype=np.float32)

# Raw frame dimensions buffer for Stage 5
local_var_buf = np.zeros((2160, 3840), dtype=np.float32)

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

    # Stage 1: Illumination Normalization (8x Downsampled Background Estimation)
    t0 = cv2.getTickCount()
    ds_factor = 8
    ds_w = w // ds_factor
    ds_h = h // ds_factor

    img_ds = cv2.resize(img, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

    blur_pad_ds = max(1, 125 // ds_factor)
    ksize_ds_stage1 = 31  

    padded_raw_img_ds = cv2.copyMakeBorder(
        img_ds, blur_pad_ds, blur_pad_ds, blur_pad_ds, blur_pad_ds, borderType=cv2.BORDER_REFLECT_101
    )
    blur_padded_ds = cv2.GaussianBlur(padded_raw_img_ds, (ksize_ds_stage1, ksize_ds_stage1), 0)
    blur_ds = blur_padded_ds[blur_pad_ds:-blur_pad_ds, blur_pad_ds:-blur_pad_ds]

    blur = cv2.resize(blur_ds, (w, h), interpolation=cv2.INTER_LINEAR)

    normalized_img = cv2.divide(img, blur, scale=128)
    t_illum_norm = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 2: 2D FFT Frequency-Domain Gabor Filtering (2x Spatial Downsampling)
    t0 = cv2.getTickCount()
    
    # Decimate normalized image to 1080p
    norm_img_ds2 = cv2.resize(normalized_img, (w // DS_STAGE2, h // DS_STAGE2), interpolation=cv2.INTER_AREA)

    # Probe 2a: Reflective Padding in 1080p domain
    t_sub0 = cv2.getTickCount()
    padded_img = cv2.copyMakeBorder(
        norm_img_ds2, 
        top=pad_size_ds, 
        bottom=pad_size_ds, 
        left=pad_size_ds, 
        right=pad_size_ds, 
        borderType=cv2.BORDER_REFLECT_101
    )
    padded_h, padded_w = padded_img.shape
    t_fft_reflect_pad = (cv2.getTickCount() - t_sub0) / freq * 1000.0

    # Probe 2b: Canvas Zero-Padding into Pre-Allocated Static Buffer
    t_sub0 = cv2.getTickCount()
    padded_img_opt_buf[:padded_h, :padded_w] = padded_img
    padded_img_opt_buf[padded_h:, :] = 0
    padded_img_opt_buf[:, padded_w:] = 0
    t_fft_zero_pad = (cv2.getTickCount() - t_sub0) / freq * 1000.0

    # Probe 2c: Forward CCS Real DFT into Static Target Buffer
    t_sub0 = cv2.getTickCount()
    cv2.dft(padded_img_opt_buf, dst=dft_img_buf, flags=cv2.DFT_REAL_OUTPUT)
    t_fft_forward = (cv2.getTickCount() - t_sub0) / freq * 1000.0

    # Probe 2d: CCS Spectral Multiplications
    t_sub0 = cv2.getTickCount()
    dft_res_real = cv2.mulSpectrums(dft_img_buf, dft_kernel_real, flags=cv2.DFT_REAL_OUTPUT)
    dft_res_imag = cv2.mulSpectrums(dft_img_buf, dft_kernel_imag, flags=cv2.DFT_REAL_OUTPUT)
    t_fft_mulspectrums = (cv2.getTickCount() - t_sub0) / freq * 1000.0

    # Probe 2e: Dual Real Inverse DFTs
    t_sub0 = cv2.getTickCount()
    idft_real = cv2.idft(dft_res_real, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    idft_imag = cv2.idft(dft_res_imag, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    t_fft_inverse = (cv2.getTickCount() - t_sub0) / freq * 1000.0

    # Probe 2f: Unpad DFT Canvas & Structural Energy Calculation
    t_sub0 = cv2.getTickCount()
    f_real_view = idft_real[:padded_h, :padded_w][pad_size_ds:-pad_size_ds, pad_size_ds:-pad_size_ds]
    f_imag_view = idft_imag[:padded_h, :padded_w][pad_size_ds:-pad_size_ds, pad_size_ds:-pad_size_ds]

    struct_energy_ds = cv2.magnitude(f_real_view, f_imag_view)

    # Upsample structural energy map back to original 4K resolution
    structural_energy = cv2.resize(struct_energy_ds, (w, h), interpolation=cv2.INTER_LINEAR)
    t_fft_magnitude = (cv2.getTickCount() - t_sub0) / freq * 1000.0

    t_gabor = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 3: Z-Score Envelope & Thresholding (Optimized via Direct Strided Slicing)
    t0 = cv2.getTickCount()
    norm_img_f32 = normalized_img.astype(np.float32)

    # Direct 8x strided slice view (Zero allocation, avoids cv2.resize memory penalty)
    struct_sliced = structural_energy[::8, :]

    # SIMD C-Accelerated Column Reductions
    col_energy_mean_1d = cv2.reduce(struct_sliced, dim=0, rtype=cv2.REDUCE_AVG, dtype=cv2.CV_32F)
    col_int_mean_1d = cv2.reduce(norm_img_f32[::8, :], dim=0, rtype=cv2.REDUCE_AVG, dtype=cv2.CV_32F)

    col_energy_mean = cv2.GaussianBlur(col_energy_mean_1d, (101, 1), 0)
    col_int_mean = cv2.GaussianBlur(col_int_mean_1d, (101, 1), 0)

    z_struct_energy = (structural_energy - col_energy_mean) * inv_struct_std
    struct_fray_mask = z_struct_energy > K_SIGMA
    struct_void_mask = z_struct_energy < -K_SIGMA
    struct_mask = struct_fray_mask | struct_void_mask

    z_intensity = (norm_img_f32 - col_int_mean) * inv_int_std
    dark_mask = z_intensity < -K_SIGMA
    light_saturation_mask = z_intensity > K_SIGMA
    intensity_mask = dark_mask | light_saturation_mask

    combined_pixel_mask = (struct_mask | intensity_mask).astype(np.uint8) * 255
    t_zscore = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 4: Structural Morphological Cleaning
    t0 = cv2.getTickCount()
    closed_mask = cv2.morphologyEx(combined_pixel_mask, cv2.MORPH_CLOSE, close_kernel)

    cleaned_mask = cv2.morphologyEx(
        closed_mask, 
        cv2.MORPH_OPEN, 
        aligned_vert_kernel, 
        anchor=(center_idx, center_idx),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    t_struct_morph = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 5: Local Variance Stream (Single-Pass uint8 Box Filters & Pre-allocated Buffers)
    t0 = cv2.getTickCount()
    mean_I = cv2.boxFilter(img, ddepth=cv2.CV_32F, ksize=(V_WIN, V_WIN))
    mean_I2 = cv2.sqrBoxFilter(img, ddepth=cv2.CV_32F, ksize=(V_WIN, V_WIN))

    cv2.subtract(mean_I2, cv2.multiply(mean_I, mean_I), dst=local_var_buf)

    _, raw_oil_mask_float = cv2.threshold(local_var_buf, var_limit, 255, cv2.THRESH_BINARY)
    raw_oil_mask = raw_oil_mask_float.astype(np.uint8)

    closed_oil_mask = cv2.morphologyEx(raw_oil_mask, cv2.MORPH_CLOSE, oil_close_kernel)
    solid_oil_mask = cv2.morphologyEx(closed_oil_mask, cv2.MORPH_OPEN, oil_open_kernel)

    unified_candidate_mask = cv2.bitwise_or(cleaned_mask, solid_oil_mask)
    t_variance_stream = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 6: Connected Components Size Filtering (Zero-Allocation 1D Array Mapping)
    t0 = cv2.getTickCount()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unified_candidate_mask)

    MIN_CLUSTER_SIZE = 270
    
    if num_labels > 1:
        areas = stats[:, cv2.CC_STAT_AREA]
        
        # Build 1D lookup vector: label_id -> 255 (if valid) or 0
        lut = np.zeros(num_labels, dtype=np.uint8)
        valid_mask = (areas >= MIN_CLUSTER_SIZE)
        valid_mask[0] = False  # Explicitly force background (label 0) to 0
        lut[valid_mask] = 255

        defect_count = int(np.count_nonzero(valid_mask))

        # Single-pass 1D index mapping across C backend (Exact logic parity)
        final_macro_mask = lut[labels]
    else:
        final_macro_mask = np.zeros_like(unified_candidate_mask)
        defect_count = 0

    t_cc_filter = (cv2.getTickCount() - t0) / freq * 1000.0

    # Stage 7: I/O Write
    t0 = cv2.getTickCount()
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, final_macro_mask)
    t_io_write = (cv2.getTickCount() - t0) / freq * 1000.0

    # Accumulate primary metrics
    stage_latencies["01_io_read"].append(t_io_read)
    stage_latencies["02_illum_norm"].append(t_illum_norm)
    stage_latencies["03_gabor_filter"].append(t_gabor)
    stage_latencies["04_zscore_envelope"].append(t_zscore)
    stage_latencies["05_struct_morph"].append(t_struct_morph)
    stage_latencies["06_variance_stream"].append(t_variance_stream)
    stage_latencies["07_cc_filtering"].append(t_cc_filter)
    stage_latencies["08_io_write"].append(t_io_write)

    # Accumulate Stage 2 sub-metrics
    stage_latencies["03a_fft_reflect_pad"].append(t_fft_reflect_pad)
    stage_latencies["03b_fft_zero_pad"].append(t_fft_zero_pad)
    stage_latencies["03c_fft_forward"].append(t_fft_forward)
    stage_latencies["03d_fft_mulspectrums"].append(t_fft_mulspectrums)
    stage_latencies["03e_fft_inverse"].append(t_fft_inverse)
    stage_latencies["03f_fft_magnitude"].append(t_fft_magnitude)

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
    
    # Sub-stage formatting
    if key.startswith(("03a", "03b", "03c", "03d", "03e", "03f")):
        stage_name = "  └─ " + key[4:].replace('_', ' ').title()
    else:
        stage_name = key[3:].replace('_', ' ').title()
        
    print(f"{stage_name:<28} | {avg_t:<10.2f} | {min_t:<10.2f} | {max_t:<10.2f} | {pct:<8.1f}")

print("-" * 65)
print(f"{'TOTAL PIPELINE (per frame)':<28} | {avg_total:<10.2f} | {np.min(stage_latencies['00_total']):<10.2f} | {np.max(stage_latencies['00_total']):<10.2f} | 100.0%")
print("=" * 65)