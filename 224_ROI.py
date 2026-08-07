import os
import glob
import json
import cv2
import numpy as np

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
INPUT_FOLDER = 'input'         # Folder containing input PNG images
OUTPUT_FOLDER = 'output_defects' # Folder to save output defect masks and ROIs
CALIB_JSON_PATH = 'calibration_metrics.json'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. Load Pre-Calculated Calibration Metrics
if not os.path.exists(CALIB_JSON_PATH):
    raise FileNotFoundError(f"Error: Calibration file '{CALIB_JSON_PATH}' not found.")

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
calib_struct_std = calib["calib_struct_std"]
calib_int_std = calib["calib_int_std"]

# Pre-build Spatial Gabor Kernels (float32 precision)
kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_32F)
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

# Global DFT Initialization & Static Buffers
pad_size = ksize
raw_padded_h = 2160 + (2 * pad_size)
raw_padded_w = 3840 + (2 * pad_size)

opt_h = cv2.getOptimalDFTSize(raw_padded_h)
opt_w = cv2.getOptimalDFTSize(raw_padded_w)

kernel_real_padded = np.zeros((opt_h, opt_w), dtype=np.float32)
kernel_imag_padded = np.zeros((opt_h, opt_w), dtype=np.float32)

kernel_real_padded[:ksize, :ksize] = kernel_real
kernel_imag_padded[:ksize, :ksize] = kernel_imag

dft_kernel_real = cv2.dft(kernel_real_padded, flags=cv2.DFT_REAL_OUTPUT)
dft_kernel_imag = cv2.dft(kernel_imag_padded, flags=cv2.DFT_REAL_OUTPUT)

inv_struct_std = np.float32(1.0 / (calib_struct_std + 1e-7))
inv_int_std = np.float32(1.0 / (calib_int_std + 1e-7))

padded_img_opt_buf = np.zeros((opt_h, opt_w), dtype=np.float32)
dft_img_buf = np.zeros((opt_h, opt_w), dtype=np.float32)
local_var_buf = np.zeros((2160, 3840), dtype=np.float32)


# =====================================================================
# LAYER 1 ANOMALY DETECTOR
# =====================================================================
def run_layer1_detector(img_path):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Skipping corrupt image: {img_path}")
        return filename, []

    h, w = img.shape

    # Stage 1: Illumination Normalization
    ds_factor = 8
    ds_w = w // ds_factor
    ds_h = h // ds_factor

    img_ds = cv2.resize(img, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

    blur_pad_ds = max(1, 125 // ds_factor)
    ksize_ds = 31  

    padded_raw_img_ds = cv2.copyMakeBorder(
        img_ds, blur_pad_ds, blur_pad_ds, blur_pad_ds, blur_pad_ds, borderType=cv2.BORDER_REFLECT_101
    )
    blur_padded_ds = cv2.GaussianBlur(padded_raw_img_ds, (ksize_ds, ksize_ds), 0)
    blur_ds = blur_padded_ds[blur_pad_ds:-blur_pad_ds, blur_pad_ds:-blur_pad_ds]

    blur = cv2.resize(blur_ds, (w, h), interpolation=cv2.INTER_LINEAR)
    normalized_img = cv2.divide(img, blur, scale=128)

    # Stage 2: 2D FFT Frequency-Domain Gabor Filtering
    padded_img = cv2.copyMakeBorder(
        normalized_img, top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, borderType=cv2.BORDER_REFLECT_101
    )
    padded_h, padded_w = padded_img.shape

    padded_img_opt_buf[:padded_h, :padded_w] = padded_img
    padded_img_opt_buf[padded_h:, :] = 0
    padded_img_opt_buf[:, padded_w:] = 0

    cv2.dft(padded_img_opt_buf, dst=dft_img_buf, flags=cv2.DFT_REAL_OUTPUT)

    dft_res_real = cv2.mulSpectrums(dft_img_buf, dft_kernel_real, flags=cv2.DFT_REAL_OUTPUT)
    dft_res_imag = cv2.mulSpectrums(dft_img_buf, dft_kernel_imag, flags=cv2.DFT_REAL_OUTPUT)

    idft_real = cv2.idft(dft_res_real, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    idft_imag = cv2.idft(dft_res_imag, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    f_real_view = idft_real[:padded_h, :padded_w][pad_size:-pad_size, pad_size:-pad_size]
    f_imag_view = idft_imag[:padded_h, :padded_w][pad_size:-pad_size, pad_size:-pad_size]

    structural_energy = cv2.magnitude(f_real_view, f_imag_view)

    # Stage 3: Z-Score Envelope & Thresholding
    norm_img_f32 = normalized_img.astype(np.float32)
    struct_sliced = structural_energy[::8, :]

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

    # Stage 4: Structural Morphological Cleaning
    closed_mask = cv2.morphologyEx(combined_pixel_mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned_mask = cv2.morphologyEx(
        closed_mask, 
        cv2.MORPH_OPEN, 
        aligned_vert_kernel, 
        anchor=(center_idx, center_idx),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Stage 5: Local Variance Stream
    mean_I = cv2.boxFilter(img, ddepth=cv2.CV_32F, ksize=(V_WIN, V_WIN))
    mean_I2 = cv2.sqrBoxFilter(img, ddepth=cv2.CV_32F, ksize=(V_WIN, V_WIN))

    cv2.subtract(mean_I2, cv2.multiply(mean_I, mean_I), dst=local_var_buf)

    _, raw_oil_mask_float = cv2.threshold(local_var_buf, var_limit, 255, cv2.THRESH_BINARY)
    raw_oil_mask = raw_oil_mask_float.astype(np.uint8)

    closed_oil_mask = cv2.morphologyEx(raw_oil_mask, cv2.MORPH_CLOSE, oil_close_kernel)
    solid_oil_mask = cv2.morphologyEx(closed_oil_mask, cv2.MORPH_OPEN, oil_open_kernel)

    unified_candidate_mask = cv2.bitwise_or(cleaned_mask, solid_oil_mask)

    # Stage 6: Connected Components Extraction (MIN_CLUSTER_SIZE = 270)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(unified_candidate_mask)

    MIN_CLUSTER_SIZE = 270
    components = []

    if num_labels > 1:
        areas = stats[:, cv2.CC_STAT_AREA]
        
        lut = np.zeros(num_labels, dtype=np.uint8)
        valid_mask = (areas >= MIN_CLUSTER_SIZE)
        valid_mask[0] = False
        lut[valid_mask] = 255

        final_macro_mask = lut[labels]

        for i in range(1, num_labels):
            if valid_mask[i]:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[i]

                components.append({
                    "id": int(i),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": int(areas[i]),
                    "cx": float(cx),
                    "cy": float(cy),
                    "covered": False
                })
    else:
        final_macro_mask = np.zeros_like(unified_candidate_mask)

    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, final_macro_mask)

    return filename, components


# =====================================================================
# STAGE 2: ROI MIDDLEWARE GENERATOR (224x224 ADAPTED SPATIAL MATH)
# =====================================================================
def generate_rois(components):
    """
    Executes Priority 1, 2, and 3 ROI Planning over Layer 1 components.
    Returns candidate bounding boxes with priority metadata:
    [(x, y, w, h, priority_str, order_num), ...]
    """
    tagged_boxes = []

    # -----------------------------------------------------------------
    # PRIORITY 1: Large Single Defects (Area > 700px, Highest Area First)
    # -----------------------------------------------------------------
    p1_components = [c for c in components if c["area"] > 700 and not c["covered"]]
    
    # Sort P1 components strictly by area descending
    p1_components.sort(key=lambda c: c["area"], reverse=True)
    
    p1_count = 0
    for comp in p1_components:
        p1_count += 1
        tagged_boxes.append((comp["x"], comp["y"], comp["w"], comp["h"], "P1", p1_count))
        comp["covered"] = True

    # Update coverage state for remaining components that fall inside P1 boxes
    _update_coverage(components, [(b[0], b[1], b[2], b[3]) for b in tagged_boxes])

    # -----------------------------------------------------------------
    # PRIORITY 2: BFS Clusters (270px <= Area <= 700px, Distance < 105px, Max Span <= 175px)
    # -----------------------------------------------------------------
    p2_pool = [c for c in components if 270 <= c["area"] <= 700 and not c["covered"]]
    
    if p2_pool:
        clusters = _bfs_spatial_clustering(p2_pool, max_dist=105.0, max_span=175)
        
        # Calculate clusterScore for ranking: clusterScore = total_area * density
        cluster_scores = []
        for cluster in clusters:
            tot_area = sum(c["area"] for c in cluster)
            min_x = min(c["x"] for c in cluster)
            min_y = min(c["y"] for c in cluster)
            max_x = max(c["x"] + c["w"] for c in cluster)
            max_y = max(c["y"] + c["h"] for c in cluster)
            bw = max(1, max_x - min_x)
            bh = max(1, max_y - min_y)
            
            density = tot_area / float(bw * bh)
            score = tot_area * density
            
            cluster_scores.append((score, cluster, min_x, min_y, bw, bh))

        # Process clusters ranked by clusterScore descending
        cluster_scores.sort(key=lambda item: item[0], reverse=True)

        p2_count = 0
        for score, cluster, cx_min, cy_min, cbw, cbh in cluster_scores:
            uncovered_members = [c for c in cluster if not c["covered"]]
            if not uncovered_members:
                continue

            p2_count += 1
            tagged_boxes.append((cx_min, cy_min, cbw, cbh, "P2", p2_count))
            for c in cluster:
                c["covered"] = True

    # Update coverage state again
    _update_coverage(components, [(b[0], b[1], b[2], b[3]) for b in tagged_boxes])

    # -----------------------------------------------------------------
    # PRIORITY 3: Near-Miss & Remaining Isolated Points (Scaled Radii)
    # -----------------------------------------------------------------
    p3_pool = [c for c in components if not c["covered"]]

    p3_count = 0
    for comp in p3_pool:
        if comp["covered"]:
            continue

        merged = False
        
        for idx, item in enumerate(tagged_boxes):
            bx, by, bw, bh, prio, num = item
            box_cx = bx + bw / 2.0
            box_cy = by + bh / 2.0
            dist = np.sqrt((comp["cx"] - box_cx)**2 + (comp["cy"] - box_cy)**2)

            new_x1 = min(bx, comp["x"])
            new_y1 = min(by, comp["y"])
            new_x2 = max(bx + bw, comp["x"] + comp["w"])
            new_y2 = max(by + bh, comp["y"] + comp["h"])
            
            span_w = new_x2 - new_x1
            span_h = new_y2 - new_y1

            # Condition 1: Centroid dist < 105px and combined span <= 175px -> Merge
            if dist < 105.0 and span_w <= 175 and span_h <= 175:
                tagged_boxes[idx] = (new_x1, new_y1, span_w, span_h, prio, num)
                comp["covered"] = True
                merged = True
                break

            # Condition 2: Centroid dist 105px-157.5px and combined span <= 175px -> Shift & Merge
            elif 105.0 <= dist <= 157.5 and span_w <= 175 and span_h <= 175:
                tagged_boxes[idx] = (new_x1, new_y1, span_w, span_h, prio, num)
                comp["covered"] = True
                merged = True
                break

        # Condition 3: Otherwise create a separate bounding box
        if not merged:
            p3_count += 1
            tagged_boxes.append((comp["x"], comp["y"], comp["w"], comp["h"], "P3", p3_count))
            comp["covered"] = True

    return tagged_boxes


def _bfs_spatial_clustering(components, max_dist=105.0, max_span=175):
    n = len(components)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        cluster = [components[i]]
        queue = [components[i]]

        while queue:
            curr = queue.pop(0)

            for j in range(n):
                if visited[j]:
                    continue

                target = components[j]
                dist = np.sqrt((curr["cx"] - target["cx"])**2 + (curr["cy"] - target["cy"])**2)

                if dist < max_dist:
                    temp_cluster = cluster + [target]
                    tx1 = min(c["x"] for c in temp_cluster)
                    ty1 = min(c["y"] for c in temp_cluster)
                    tx2 = max(c["x"] + c["w"] for c in temp_cluster)
                    ty2 = max(c["y"] + c["h"] for c in temp_cluster)

                    if (tx2 - tx1) <= max_span and (ty2 - ty1) <= max_span:
                        visited[j] = True
                        cluster.append(target)
                        queue.append(target)

        clusters.append(cluster)

    return clusters


def _update_coverage(components, boxes):
    for comp in components:
        if comp["covered"]:
            continue
        cx1, cy1 = comp["x"], comp["y"]
        cx2, cy2 = comp["x"] + comp["w"], comp["y"] + comp["h"]

        for (bx, by, bw, bh) in boxes:
            if cx1 >= bx and cy1 >= by and cx2 <= (bx + bw) and cy2 <= (by + bh):
                comp["covered"] = True
                break


# =====================================================================
# STAGE 3: CROP & NORMALIZE ENGINE (NATIVE 1:1 SCALE 224x224 RGB)
# =====================================================================
def extract_and_save_rois(img_path, tagged_boxes, output_folder):
    raw_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if raw_rgb is None:
        return 0

    img_h, img_w = raw_rgb.shape[:2]
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]

    # 1. Centroid-Distance NMS (< 112px center distance suppression)
    deduped_boxes = []
    for item in tagged_boxes:
        bx, by, bw, bh, prio, num = item
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        
        keep = True
        for d_item in deduped_boxes:
            dx, dy, dw, dh = d_item[:4]
            dcx, dcy = dx + dw / 2.0, dy + dh / 2.0
            dist = np.sqrt((bcx - dcx)**2 + (bcy - dcy)**2)
            if dist < 112.0:
                keep = False
                break
        if keep:
            deduped_boxes.append(item)

    roi_count = 0

    # 2. Extract Patches (Native 224x224 Scale)
    for item in deduped_boxes:
        bx, by, bw, bh, prio, num = item

        if bw <= 224 and bh <= 224:
            bcx, bcy = int(bx + bw // 2), int(by + bh // 2)

            x1 = bcx - 112
            y1 = bcy - 112
            x2 = x1 + 224
            y2 = y1 + 224

            if x1 < 0:
                x2 += -x1
                x1 = 0
            if y1 < 0:
                y2 += -y1
                y1 = 0
            if x2 > img_w:
                x1 -= (x2 - img_w)
                x2 = img_w
            if y2 > img_h:
                y1 -= (y2 - img_h)
                y2 = img_h

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            crop_rgb = raw_rgb[y1:y2, x1:x2]

            if crop_rgb.shape[0] != 224 or crop_rgb.shape[1] != 224:
                crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)

            roi_count += 1
            save_path = os.path.join(output_folder, f"{base_name}_roi_{prio}_{num}.png")
            cv2.imwrite(save_path, crop_rgb)

        else:
            stride = 112
            tile_sub = 0
            for ty in range(by, by + bh, stride):
                for tx in range(bx, bx + bw, stride):
                    tile_sub += 1
                    x1 = tx
                    y1 = ty
                    x2 = min(img_w, x1 + 224)
                    y2 = min(img_h, y1 + 224)

                    if (x2 - x1) < 224 and x2 == img_w:
                        x1 = max(0, img_w - 224)
                    if (y2 - y1) < 224 and y2 == img_h:
                        y1 = max(0, img_h - 224)

                    crop_rgb = raw_rgb[y1:y2, x1:x2]

                    if crop_rgb.shape[0] != 224 or crop_rgb.shape[1] != 224:
                        crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)

                    roi_count += 1
                    save_path = os.path.join(output_folder, f"{base_name}_roi_{prio}_{num}_tile{tile_sub}.png")
                    cv2.imwrite(save_path, crop_rgb)

    return roi_count


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.png'))

    if not image_paths:
        print(f"No PNG files found in '{INPUT_FOLDER}'.")
        exit()

    print(f"Processing {len(image_paths)} images through Layer 1 & ROI Middleware (224x224 native)...\n")

    for img_path in image_paths:
        # Layer 1 Detection
        filename, components = run_layer1_detector(img_path)

        if not components:
            print(f"[PASS] {filename} -> No defects flagged.")
            continue

        # Stage 2: Priority ROI Planning
        tagged_boxes = generate_rois(components)

        # Stage 3: Raw RGB Crop & Save Engine
        num_rois_saved = extract_and_save_rois(img_path, tagged_boxes, OUTPUT_FOLDER)

        print(f"[FAIL] {filename} -> Flagged Components: {len(components)} | Generated 224x224 ROIs: {num_rois_saved}")