import os
import json
import time
import csv
import cv2
import av
import numpy as np

# =====================================================================
# GLOBAL CONFIGURATION (SINGLE SOURCE OF TRUTH)
# =====================================================================
DEFAULT_VIDEO_PATH = "asset/cuts/c1.MOV"
DEFAULT_CALIB_JSON = "calibration_metrics.json"
DEFAULT_EVAL_DIR = "evaluation_results"
DEFAULT_SAMPLE_INTERVAL = 1.0

def log_debug(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [DEBUG] {msg}", flush=True)


# =====================================================================
# METRICS & PIPELINE HELPER FUNCTIONS
# =====================================================================

def compute_laplacian_variance(gray_frame, roi_box=None):
    if roi_box:
        x, y, w, h = roi_box
        gray_frame = gray_frame[y:y+h, x:x+w]
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var()

def compute_nwer(gray_img, calib):
    """
    Computes Normalized Weave Energy Retention (NWER) using calibrated normalized peak coordinates
    normalized against Total AC Spectral Energy (excluding central DC component).
    """
    h, w = gray_img.shape
    cy, cx = h // 2, w // 2

    # 1. 2D FFT Shift & Power Spectrum Calculation
    f = np.fft.fft2(gray_img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag_sq = np.abs(fshift) ** 2

    y, x = np.ogrid[:h, :w]

    # 2. Total AC Energy (Exclude DC spike r <= 3)
    ac_mask = np.sqrt((x - cx)**2 + (y - cy)**2) > 3
    E_ac = np.sum(mag_sq[ac_mask]) + 1e-7

    # 3. Dynamic Mask Generation using Calibrated Peak Coordinates
    if "peak_coords_normalized" in calib:
        weave_mask = np.zeros((h, w), dtype=bool)
        radius = calib.get("peak_radius_px", 5)

        for norm_y, norm_x in calib["peak_coords_normalized"]:
            py = int(cy + norm_y * h)
            px = int(cx + norm_x * w)
            p_dist = np.sqrt((x - px)**2 + (y - py)**2)
            weave_mask[p_dist <= radius] = True
    else:
        # Fallback annular ring if peak coordinates are missing
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        weave_mask = (dist_from_center >= 15) & (dist_from_center <= 80)

    # 4. Weave Band Energy & NWER Ratio Calculation
    E_weave = np.sum(mag_sq[weave_mask])
    nwer = float(E_weave / E_ac)

    return round(nwer, 5)

def build_pipeline_kernels(calib):
    ksize = calib["ksize"]
    sigma = calib["sigma"]
    theta = calib["theta"]
    lambd = calib["lambd"]
    gamma = calib["gamma"]

    kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
    kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
    kernel_real -= kernel_real.mean()
    kernel_imag -= kernel_imag.mean()

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    LINE_LEN = 15
    base_v_kernel = np.zeros((LINE_LEN, LINE_LEN), dtype=np.uint8)
    base_v_kernel[:, LINE_LEN // 2] = 1 
    theta_deg = np.degrees(theta) + 90.0
    M_rot = cv2.getRotationMatrix2D((LINE_LEN / 2.0, LINE_LEN / 2.0), theta_deg, 1.0)
    aligned_vert_kernel = cv2.warpAffine(base_v_kernel, M_rot, (LINE_LEN, LINE_LEN), flags=cv2.INTER_NEAREST)

    oil_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    oil_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    return {
        "kernel_real": kernel_real,
        "kernel_imag": kernel_imag,
        "close_kernel": close_kernel,
        "aligned_vert_kernel": aligned_vert_kernel,
        "oil_close_kernel": oil_close_kernel,
        "oil_open_kernel": oil_open_kernel
    }


def detect_defects(gray_img, calib, kernels):
    ksize = calib["ksize"]
    K_SIGMA = calib["K_SIGMA"]
    V_WIN = calib["V_WIN"]
    var_limit = calib["var_limit"]

    calib_struct_std = calib["calib_struct_std"]
    calib_int_std = calib["calib_int_std"]

    # 1. Illumination Normalization
    blur_pad = 125
    padded_raw = cv2.copyMakeBorder(gray_img, blur_pad, blur_pad, blur_pad, blur_pad, borderType=cv2.BORDER_REFLECT_101)
    blur_padded = cv2.GaussianBlur(padded_raw, (251, 251), 0)
    blur = blur_padded[blur_pad:-blur_pad, blur_pad:-blur_pad]
    normalized_img = cv2.divide(gray_img.astype(np.float32), blur.astype(np.float32), scale=128)
    normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)
    # normalized_img = cv2.divide(gray_img, blur, scale=128)

    # 2. Reflective Padding & Gabor Energy
    pad_size = ksize
    padded_img = cv2.copyMakeBorder(normalized_img, pad_size, pad_size, pad_size, pad_size, borderType=cv2.BORDER_REFLECT_101)

    f_real_padded = cv2.filter2D(padded_img, cv2.CV_64F, kernels["kernel_real"])
    f_imag_padded = cv2.filter2D(padded_img, cv2.CV_64F, kernels["kernel_imag"])

    f_real = f_real_padded[pad_size:-pad_size, pad_size:-pad_size]
    f_imag = f_imag_padded[pad_size:-pad_size, pad_size:-pad_size]
    structural_energy = np.sqrt(f_real**2 + f_imag**2)

    # 3. Z-Score Envelope
    col_energy_mean = cv2.GaussianBlur(np.mean(structural_energy, axis=0, keepdims=True), (101, 1), 0)
    col_int_mean = cv2.GaussianBlur(np.mean(normalized_img.astype(np.float32), axis=0, keepdims=True), (101, 1), 0)

    z_struct_energy = (structural_energy - col_energy_mean) / (calib_struct_std + 1e-7)
    struct_mask = (z_struct_energy > K_SIGMA) | (z_struct_energy < -K_SIGMA)

    z_intensity = (normalized_img.astype(np.float32) - col_int_mean) / (calib_int_std + 1e-7)
    intensity_mask = (z_intensity < -K_SIGMA) | (z_intensity > K_SIGMA)

    combined_pixel_mask = (struct_mask | intensity_mask).astype(np.uint8) * 255
    closed_mask = cv2.morphologyEx(combined_pixel_mask, cv2.MORPH_CLOSE, kernels["close_kernel"])
    cleaned_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernels["aligned_vert_kernel"])

   # 4. Local Variance Stream (Oil Marks)
    gray_calc = gray_img.copy().astype(np.float32)
    mean_I = cv2.blur(gray_calc, (V_WIN, V_WIN))
    mean_I2 = cv2.blur(gray_calc**2, (V_WIN, V_WIN))
    local_var = np.maximum(0, mean_I2 - (mean_I ** 2))  # Clamp negative floating-point noise to 0

    _, raw_oil_mask_float = cv2.threshold(local_var, var_limit, 255, cv2.THRESH_BINARY)
    raw_oil_mask = np.clip(raw_oil_mask_float, 0, 255).astype(np.uint8)  # Explicitly bound array values

    closed_oil_mask = cv2.morphologyEx(raw_oil_mask, cv2.MORPH_CLOSE, kernels["oil_close_kernel"])
    solid_oil_mask = cv2.morphologyEx(closed_oil_mask, cv2.MORPH_OPEN, kernels["oil_open_kernel"])

    unified_candidate_mask = cv2.bitwise_or(cleaned_mask, solid_oil_mask)

    # 5. Connected Component Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unified_candidate_mask)
    MIN_CLUSTER_SIZE = 120
    final_macro_mask = np.zeros_like(unified_candidate_mask)
    defect_count = 0

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_CLUSTER_SIZE:
            final_macro_mask[labels == i] = 255
            defect_count += 1

    return final_macro_mask, defect_count


# =====================================================================
# INTERACTIVE REVIEW UI RENDERING
# =====================================================================

def render_ui(img_bgr, mask, overlay, metrics_dict, user_state, dashboard_height=None):
    """
    Renders UI Dashboard with dynamic font scaling proportional to input image resolution.
    """
    h_img, w_img, _ = img_bgr.shape
    total_w = w_img * 2
    
    # Calculate scale factor based on width (Reference: 1280px width = 1.0 scale)
    scale_factor = max(0.8, total_w / 1280.0)
    
    # Scale dashboard height dynamically if not explicitly overridden
    if dashboard_height is None:
        dashboard_height = int(220 * scale_factor)
        
    dashboard = np.zeros((dashboard_height, total_w, 3), dtype=np.uint8)
    
    # Scale fonts and line positions dynamically
    H_SCALE = 1.1 * scale_factor
    T_SCALE = 0.9 * scale_factor
    THICKNESS = max(2, int(2.5 * scale_factor))
    
    col1_x = int(30 * scale_factor)
    col2_x = int(total_w * 0.32)
    col3_x = int(total_w * 0.65)
    
    y_line1 = int(50 * scale_factor)
    y_line2 = int(110 * scale_factor)
    y_line3 = int(165 * scale_factor)
    
    # --- Column 1: Calculated Metrics ---
    cv2.putText(dashboard, "METRICS:", (col1_x, y_line1), cv2.FONT_HERSHEY_SIMPLEX, H_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
    nwer_val = metrics_dict.get('nwer', 'N/A')
    cv2.putText(dashboard, f"NWER: {nwer_val}", (col1_x, y_line2), cv2.FONT_HERSHEY_SIMPLEX, T_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)
    
    # --- Column 2: User Inputs ---
    cv2.putText(dashboard, "MANUAL INPUTS:", (col2_x, y_line1), cv2.FONT_HERSHEY_SIMPLEX, H_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
    
    det_str = user_state['detected'] if user_state['detected'] is not None else "_"
    det_color = (0, 255, 0) if det_str == "No" else ((0, 0, 255) if det_str == "Yes" else (200, 200, 200))
    cv2.putText(dashboard, f"Defect Detected: {det_str}", (col2_x, y_line2), cv2.FONT_HERSHEY_SIMPLEX, T_SCALE, det_color, THICKNESS, cv2.LINE_AA)
    
    noise_str = user_state['noise'] if user_state['noise'] is not None else "_"
    cv2.putText(dashboard, f"Noise Rating   : {noise_str}", (col2_x, y_line3), cv2.FONT_HERSHEY_SIMPLEX, T_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)
    
    # --- Column 3: Controls ---
    cv2.putText(dashboard, "CONTROLS:", (col3_x, y_line1), cv2.FONT_HERSHEY_SIMPLEX, H_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
    cv2.putText(dashboard, "Y/N/S : Defect (Yes/No/Skip)", (col3_x, int(85 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, T_SCALE * 0.85, (200, 200, 200), max(1, THICKNESS - 1), cv2.LINE_AA)
    cv2.putText(dashboard, "0-3   : Noise Rating (0-3)", (col3_x, int(115 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, T_SCALE * 0.85, (200, 200, 200), max(1, THICKNESS - 1), cv2.LINE_AA)
    cv2.putText(dashboard, "R     : Reset Inputs", (col3_x, int(145 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, T_SCALE * 0.85, (0, 255, 255), max(1, THICKNESS - 1), cv2.LINE_AA)
    cv2.putText(dashboard, "SPACE : Confirm & Save", (col3_x, int(180 * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, T_SCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)

    # Stack composite top view and dashboard bottom view
    composite_view = np.hstack([img_bgr, overlay])
    full_ui = np.vstack([composite_view, dashboard])
    
    return full_ui


# =====================================================================
# SINGLE VIDEO EVALUATION PIPELINE
# =====================================================================

def run_single_video_evaluation(
    video_path=None,
    calib_json_path=None,
    eval_output_dir=None,
    sample_interval_sec=None,
    roi_box=None
):
    video_path = video_path or DEFAULT_VIDEO_PATH
    calib_json_path = calib_json_path or DEFAULT_CALIB_JSON
    eval_output_dir = eval_output_dir or DEFAULT_EVAL_DIR
    sample_interval_sec = sample_interval_sec if sample_interval_sec is not None else DEFAULT_SAMPLE_INTERVAL
    log_debug(f"Starting single-video evaluation session for: {video_path}")

    # Step 1: Target Video Verification
    if not os.path.exists(video_path):
        log_debug(f"[FATAL] Target video file does not exist: {video_path}")
        return

    # Step 2: Calibration Verification
    if not os.path.exists(calib_json_path):
        log_debug(f"[FATAL] Calibration metrics file missing: {calib_json_path}")
        return

    with open(calib_json_path, "r") as f:
        calib = json.load(f)
    log_debug("Calibration loaded successfully.")

    log_debug("Building pipeline kernels...")
    kernels = build_pipeline_kernels(calib)
    log_debug("Kernels created successfully.")

    # Step 3: Setup Output Directories & CSV Setup
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    defect_type = os.path.basename(os.path.dirname(video_path))

    # Dedicated folder per Video_ID
    frames_out_dir = os.path.join(eval_output_dir, "saved_frames", video_id)
    os.makedirs(frames_out_dir, exist_ok=True)
    csv_path = os.path.join(eval_output_dir, "layer1_evaluation_metrics.csv")

    csv_fields = [
        "Video_ID", "Frame_Number", "Defect_Type", "Blur_Score",
        "Processing_Time_ms", "Connected_Component_Count", "Total_White_Pixels",
        "Defect_Detected", "Noise_Rating", "NWER",
        "Composite_Filename"
    ]

    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, mode="a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()


    log_debug(f"Opening PyAV Video Stream: {video_path}")
    try:
        container = av.open(video_path)
    except Exception as e:
        log_debug(f"[FATAL] PyAV failed to open video file: {e}")
        csv_file.close()
        return

    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    fps = float(stream.average_rate) if stream.average_rate else 30.0
    time_base = float(stream.time_base)
    log_debug(f"PyAV Stream opened | Codec: {stream.codec_context.name} | PixFmt: {stream.pix_fmt} | FPS: {fps:.2f}")

    frame_idx = 0
    next_target_time_sec = 0.0
    window_created = False

    # Step 5: Chopping, Processing & Manual Input Loop
    for frame in container.decode(video=0):
        timestamp_sec = float(frame.pts * time_base) if frame.pts is not None else frame_idx / fps

        # Skip frames until current timestamp reaches or exceeds target interval
        if timestamp_sec < next_target_time_sec:
            continue

        log_debug(f"--- Processing Frame {frame_idx} (Time: {timestamp_sec:.2f}s) ---")

        # Convert 10-bit HDR/Log yuv420p10le to 8-bit BGR ndarray via PyAV
        img_rgb = frame.to_ndarray(format="rgb24")
        img_bgr = np.ascontiguousarray(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Automated Metrics Calculation (Static Pre-detection Metrics)
        blur_score = round(float(compute_laplacian_variance(gray, roi_box=roi_box)), 2)
        nwer = compute_nwer(gray, calib)

        # Pure Filter Processing Time (Strictly algorithm execution, zero I/O)
        t0 = time.perf_counter()
        mask, cc_count = detect_defects(gray, calib, kernels)
        t1 = time.perf_counter()

        proc_time_ms = round((t1 - t0) * 1000.0, 2)
        log_debug(f"Filters applied in {proc_time_ms:.2f}ms | CC Count: {cc_count} | NWER: {nwer}")

        white_pixels = int(np.count_nonzero(mask))

        overlay = img_bgr.copy()
        overlay[mask == 255] = (0, 0, 255)

        metadata = {
            "video_id": video_id,
            "frame_num": frame_idx,
            "defect_type": defect_type,
            "blur_score": blur_score,
            "nwer": nwer,
            "proc_time_ms": proc_time_ms,
            "cc_count": cc_count,
            "white_pixels": white_pixels
        }

        user_state = {"detected": None, "noise": None}

        if not window_created:
            log_debug("Initializing Resizable OpenCV Interactive GUI Window...")
            cv2.namedWindow("Layer 1 Characterization UI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Layer 1 Characterization UI", 1280, 720)
            window_created = True

        log_debug("Awaiting user manual input in GUI window...")
        
        # Step 6: Interactive Review GUI Loop
        while True:
            ui_view = render_ui(img_bgr, mask, overlay, metadata, user_state)
            cv2.imshow("Layer 1 Characterization UI", ui_view)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                log_debug("User pressed 'q'. Aborting evaluation loop.")
                container.close()
                cv2.destroyAllWindows()
                csv_file.close()
                return

            elif key in (ord('r'), ord('R')):
                user_state['detected'] = None
                user_state['noise'] = None
                log_debug("User pressed 'R'. Reset manual inputs.")

            elif key == ord('y'):
                user_state['detected'] = "Yes"
            elif key == ord('n'):
                user_state['detected'] = "No"
            elif key == ord('s'):
                user_state['detected'] = "Skip"

            # Keys 0-3 set Noise Rating (0, 1, 2, 3)
            elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                user_state['noise'] = key - ord('0')


            elif key == 32:  # SPACEBAR
                if user_state['detected'] is None or user_state['noise'] is None:
                    log_debug(f"SPACE blocked due to incomplete inputs: {user_state}")
                    continue

                log_debug("Inputs complete. Writing frame composite output & appending to CSV...")
                
                # Create composite view (RGB Image + Red Defect Overlay)
                composite_view = np.hstack([img_bgr, overlay]).copy()

                # Burn enlarged metadata banner with manual inputs at bottom-left of saved image
                label_text = f"Defect: {user_state['detected']} | Noise: {user_state['noise']} | Frame: {frame_idx} ({timestamp_sec:.2f}s)"
                
                # Scaled font scale from 0.6 -> 1.0 and thickness 1 -> 2
                font_scale = 3.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Adjusted padding rectangle for larger text box
                cv2.rectangle(composite_view, (10, composite_view.shape[0] - th - 30), (30 + tw, composite_view.shape[0] - 5), (0, 0, 0), -1)
                cv2.putText(composite_view, label_text, (20, composite_view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

                base_fname = f"{video_id}_frame_{frame_idx}_sec_{timestamp_sec:.2f}"
                composite_filename = f"{base_fname}_composite.png"

                cv2.imwrite(os.path.join(frames_out_dir, composite_filename), composite_view)

                record = {
                    "Video_ID": video_id,
                    "Frame_Number": frame_idx,
                    "Defect_Type": defect_type,
                    "Blur_Score": blur_score,
                    "Processing_Time_ms": proc_time_ms,
                    "Connected_Component_Count": cc_count,
                    "Total_White_Pixels": white_pixels,
                    "Defect_Detected": user_state['detected'],
                    "Noise_Rating": user_state['noise'],
                    "NWER": nwer,
                    "Composite_Filename": composite_filename
                }
                writer.writerow(record)
                csv_file.flush()
                log_debug(f"Record successfully written to {csv_path}")
                break

        next_target_time_sec = timestamp_sec + sample_interval_sec
        frame_idx += 1

    container.close()
    cv2.destroyAllWindows()
    csv_file.close()
    log_debug("Single video evaluation successfully completed.")
    


if __name__ == "__main__":
    run_single_video_evaluation()