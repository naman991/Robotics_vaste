import os
import json
import av
import cv2
import numpy as np

def compute_laplacian_variance(gray_frame, roi_box=None):
    if roi_box:
        x, y, w, h = roi_box
        gray_frame = gray_frame[y:y+h, x:x+w]
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var()


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
    normalized_img = cv2.divide(gray_img, blur, scale=128)

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
    img_f = gray_img.astype(np.float32)
    mean_I = cv2.blur(img_f, (V_WIN, V_WIN))
    mean_I2 = cv2.blur(img_f**2, (V_WIN, V_WIN))
    local_var = mean_I2 - (mean_I ** 2)

    _, raw_oil_mask_float = cv2.threshold(local_var, var_limit, 255, cv2.THRESH_BINARY)
    raw_oil_mask = raw_oil_mask_float.astype(np.uint8)

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


def run_video_defect_detection(
    video_path,
    calib_json_path="calibration_metrics.json",
    output_dir="output_defects",
    blur_threshold=35.0,
    sample_interval_sec=1.0,
    roi_box=None
):
    if not os.path.exists(calib_json_path):
        raise FileNotFoundError(f"Calibration file missing: {calib_json_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(calib_json_path, "r") as f:
        calib = json.load(f)

    kernels = build_pipeline_kernels(calib)

    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"[FATAL] Failed to open video: {e}")
        return

    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = float(stream.average_rate)
    time_base = float(stream.time_base)

    next_target_time_sec = 0.0

    for frame_idx, frame in enumerate(container.decode(video=0)):
        timestamp_sec = float(frame.pts * time_base) if frame.pts is not None else frame_idx / fps

        if timestamp_sec < next_target_time_sec:
            continue

        img_rgb = frame.to_ndarray(format="rgb24")
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        sharpness_score = compute_laplacian_variance(gray, roi_box=roi_box)
        is_blur = sharpness_score < blur_threshold

        # Run defect detection on ALL sampled frames (including blurred)
        defect_mask, defect_count = detect_defects(gray, calib, kernels)

        status = "FAIL" if defect_count > 0 else "PASS"
        blur_tag = "BLUR" if is_blur else "SHARP"
        
        out_filename = f"mask_sec_{timestamp_sec:.2f}_{blur_tag}_{status}_defects_{defect_count}.png"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, defect_mask)

        print(f"[{status} | {blur_tag}] Frame @ {timestamp_sec:.2f}s | Score: {sharpness_score:.2f} | Defects: {defect_count} -> Saved: {out_filename}")

        next_target_time_sec = timestamp_sec + sample_interval_sec

    container.close()


if __name__ == "__main__":
    run_video_defect_detection(
        video_path="asset/wrinkles/w1.mov",
        calib_json_path="calibration_metrics.json",
        output_dir="output_defects",
        blur_threshold=35.0,
        sample_interval_sec=1.0,
        roi_box=None
    )