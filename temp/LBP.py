"""
AITEX LBP Anomaly Detection - First Test Script
===============================================
1. Load one good image and one defect image
2. Find fabric boundary (removes left white padding)
3. Build reference histogram from good fabric
4. Scan defect image, compare each window to reference
5. Visualize results as heatmap overlay
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# ============================================================================
# CONFIGURATION - Change these paths to your actual files
# ============================================================================

GOOD_IMAGE_PATH = "dataset/good.png"
BAD_IMAGE_PATH  = "dataset/bad.png"

# LBP parameters
P = 8           # Number of neighbours
R = 1           # Radius in pixels
METHOD = 'uniform'  # 'uniform' gives 10 bins (P+2)

# Window parameters
WINDOW_SIZE = 16
STEP = 8      # 50% overlap

# Threshold (will be computed from good fabric)
CHI2_THRESHOLD = None  # Set to None for auto-computation

# Visualization
OUTPUT_HEATMAP = "anomaly_heatmap.png"


# ============================================================================
# STEP 1: Load images
# ============================================================================

def load_image(path):
    """Load image as grayscale numpy array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    print(f"Loaded {path}: shape={img.shape}, dtype={img.dtype}")
    return img


# ============================================================================
# STEP 2: Find fabric boundary
# ============================================================================

def find_fabric_start(image, white_threshold=250):
    """
    Find the column where fabric begins (white padding ends).
    Assumes white padding is on the LEFT side.
    Returns the first column index where mean intensity drops below threshold.
    """
    col_means = np.mean(image, axis=0)  # Average each column
    
    for col in range(len(col_means)):
        if col_means[col] < white_threshold:
            return col
    
    return 0  # No white padding found, whole image is fabric


def extract_fabric(image):
    """
    Remove left white padding and return only the fabric region.
    Also returns the start column for reference.
    """
    start_col = find_fabric_start(image)
    fabric = image[:, start_col:]
    print(f"Fabric extracted: start_col={start_col}, fabric_shape={fabric.shape}")
    return fabric, start_col


# ============================================================================
# STEP 3: LBP computation
# ============================================================================

def compute_lbp(image, P, R, method='uniform'):
    """
    Compute LBP codes for every pixel.
    Returns an array of integer bin labels (0 to P+1).
    For 'uniform' method with P=8: bins 0-8 are uniform patterns, bin 9 is non-uniform.
    """
    lbp = local_binary_pattern(image, P, R, method=method)
    return lbp.astype(np.uint8)


# ============================================================================
# STEP 4: Histogram operations
# ============================================================================

def build_histogram(lbp_patch, num_bins):
    """Build normalized histogram from LBP patch."""
    hist, _ = np.histogram(lbp_patch, bins=num_bins, range=(0, num_bins-1), density=True)
    return hist


def chi2_distance(hist1, hist2, epsilon=1e-6):
    """
    Chi-square distance between two normalized histograms.
    d = sum( (h1[i] - h2[i])^2 / (h1[i] + h2[i] + epsilon) )
    """
    numerator = (hist1 - hist2) ** 2
    denominator = hist1 + hist2 + epsilon
    return np.sum(numerator / denominator)


# ============================================================================
# STEP 5: Build reference histogram from good fabric
# ============================================================================

def build_reference(fabric_image, P, R, window_size, step):
    """
    Slide windows over good fabric, compute LBP histograms,
    return the average histogram and the chi2 distances of all normal windows.
    """
    num_bins = P + 2  # For uniform LBP
    lbp_full = compute_lbp(fabric_image, P, R, METHOD)
    
    h, w = lbp_full.shape
    all_histograms = []
    
    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            patch = lbp_full[y:y+window_size, x:x+window_size]
            hist = build_histogram(patch, num_bins)
            all_histograms.append(hist)
    
    all_histograms = np.array(all_histograms)
    reference = np.mean(all_histograms, axis=0)
    
    print(f"Reference built from {len(all_histograms)} windows")
    print(f"Reference histogram: {reference}")
    
    return reference, all_histograms, lbp_full


# ============================================================================
# STEP 6: Scan for anomalies
# ============================================================================

def scan_anomalies(fabric_image, reference, P, R, window_size, step, threshold):
    """
    Slide windows over fabric, compute chi2 distance to reference for each window.
    Returns anomaly scores as a 2D heatmap and list of flagged window coordinates.
    """
    num_bins = P + 2
    lbp_full = compute_lbp(fabric_image, P, R, METHOD)
    
    h, w = lbp_full.shape
    
    # Heatmap dimensions (number of window positions in y and x)
    heatmap_rows = (h - window_size) // step + 1
    heatmap_cols = (w - window_size) // step + 1
    heatmap = np.zeros((heatmap_rows, heatmap_cols))
    
    flagged_windows = []
    
    for yi, y in enumerate(range(0, h - window_size + 1, step)):
        for xi, x in enumerate(range(0, w - window_size + 1, step)):
            patch = lbp_full[y:y+window_size, x:x+window_size]
            hist = build_histogram(patch, num_bins)
            dist = chi2_distance(hist, reference)
            heatmap[yi, xi] = dist
            
            if dist > threshold:
                flagged_windows.append({
                    'x': x, 'y': y,
                    'xi': xi, 'yi': yi,
                    'distance': dist
                })
    
    print(f"Scanned {heatmap_rows * heatmap_cols} windows")
    print(f"Flagged {len(flagged_windows)} windows (threshold={threshold:.4f})")
    
    return heatmap, flagged_windows, lbp_full


# ============================================================================
# STEP 7: Auto-compute threshold from good fabric
# ============================================================================

def compute_threshold(normal_distances, sigma_multiplier=3.0):
    """
    Compute threshold from the chi2 distances of normal fabric windows.
    threshold = mean + sigma_multiplier * std
    """
    mean_dist = np.mean(normal_distances)
    std_dist = np.std(normal_distances)
    threshold = mean_dist + sigma_multiplier * std_dist
    
    print(f"Normal distances: mean={mean_dist:.6f}, std={std_dist:.6f}")
    print(f"Threshold (mean + {sigma_multiplier}*std): {threshold:.6f}")
    
    return threshold


# ============================================================================
# STEP 8: Visualization
# ============================================================================

def visualize_results(good_fabric, bad_fabric, bad_lbp, heatmap, 
                      reference_hist, flagged_windows, threshold,
                      window_size, step, output_path):
    """
    Create a 4-panel visualization:
    1. Good fabric
    2. Bad fabric with flagged windows
    3. Anomaly heatmap
    4. Reference histogram
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: Good fabric
    axes[0, 0].imshow(good_fabric, cmap='gray')
    axes[0, 0].set_title('Good Fabric (Reference)', fontsize=12)
    axes[0, 0].axis('off')
    
    # Panel 2: Bad fabric with flagged windows overlaid
    axes[0, 1].imshow(bad_fabric, cmap='gray')
    for window in flagged_windows:
        rect = plt.Rectangle((window['x'], window['y']), 
                             window_size, window_size,
                             linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(f'Defect Image - {len(flagged_windows)} windows flagged', fontsize=12)
    axes[0, 1].axis('off')
    
    # Panel 3: Anomaly heatmap (upscaled to image size for overlay)
    heatmap_display = np.zeros_like(bad_fabric, dtype=float)
    h, w = bad_fabric.shape
    for yi in range(heatmap.shape[0]):
        for xi in range(heatmap.shape[1]):
            y_px = yi * step
            x_px = xi * step
            heatmap_display[y_px:y_px+step, x_px:x_px+step] = heatmap[yi, xi]
    
    im = axes[1, 0].imshow(heatmap_display, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title(f'Anomaly Heatmap (chi2 distance)\nThreshold: {threshold:.4f}', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Panel 4: Reference histogram
    bins = np.arange(len(reference_hist))
    axes[1, 1].bar(bins, reference_hist, color='steelblue', edgecolor='black')
    axes[1, 1].set_title('Reference LBP Histogram (10 bins)', fontsize=12)
    axes[1, 1].set_xlabel('LBP Uniform Bin')
    axes[1, 1].set_ylabel('Normalized Frequency')
    axes[1, 1].set_xticks(bins)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("LBP ANOMALY DETECTION - FIRST TEST RUN")
    print("=" * 60)
    
    # --- Load ---
    print("\n[1/6] Loading images...")
    good_img = load_image(GOOD_IMAGE_PATH)
    bad_img = load_image(BAD_IMAGE_PATH)
    
    # --- Extract fabric ---
    print("\n[2/6] Extracting fabric regions...")
    good_fabric, good_start = extract_fabric(good_img)
    bad_fabric, bad_start = extract_fabric(bad_img)
    
        # --- Extract fabric ---
    print("\n[2/6] Extracting fabric regions...")
    good_fabric, good_start = extract_fabric(good_img)
    bad_fabric, bad_start = extract_fabric(bad_img)
    
    # --- FABRIC COMPARISON ---
    print("\n[FABRIC COMPARISON]")
    print(f"Good fabric: mean={good_fabric.mean():.1f}, std={good_fabric.std():.1f}")
    print(f"Bad fabric:  mean={bad_fabric.mean():.1f}, std={bad_fabric.std():.1f}")
    print(f"Good fabric: min={good_fabric.min()}, max={good_fabric.max()}")
    print(f"Bad fabric:  min={bad_fabric.min()}, max={bad_fabric.max()}")
    
    # --- Build reference ---
    print("\n[3/6] Building reference from good fabric...")
    reference_hist, normal_hists, _ = build_reference(good_fabric, P, R, WINDOW_SIZE, STEP)
    
    # --- Build reference ---
    print("\n[3/6] Building reference from good fabric...")
    reference_hist, normal_hists, _ = build_reference(good_fabric, P, R, WINDOW_SIZE, STEP)
    
    # --- Compute threshold ---
    print("\n[4/6] Computing threshold...")
    normal_distances = [chi2_distance(h, reference_hist) for h in normal_hists]
    threshold = compute_threshold(normal_distances, sigma_multiplier=3.0)
    
    # --- DIAGNOSTIC ---
    print("\n[DIAGNOSTIC]")
    print(f"Normal distance range: {min(normal_distances):.6f} to {max(normal_distances):.6f}")
    print(f"Computed threshold: {threshold:.6f}")

    # Now compute distances on the bad image to compare
    print("\nBad image distance stats:")
    bad_lbp_full = compute_lbp(bad_fabric, P, R, METHOD)
    bad_distances = []
    h, w = bad_lbp_full.shape
    num_bins = P + 2
    for y in range(0, h - WINDOW_SIZE + 1, STEP):
        for x in range(0, w - WINDOW_SIZE + 1, STEP):
            patch = bad_lbp_full[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
            hist = build_histogram(patch, num_bins)
            dist = chi2_distance(hist, reference_hist)
            bad_distances.append(dist)
    
    bad_distances = np.array(bad_distances)
    print(f"Bad image distances: min={bad_distances.min():.6f}, max={bad_distances.max():.6f}")
    print(f"Bad image distances: mean={bad_distances.mean():.6f}, std={bad_distances.std():.6f}")
    print(f"Fraction flagged at current threshold: {np.sum(bad_distances > threshold) / len(bad_distances):.2%}")

    # --- Scan defect image ---
    print("\n[5/6] Scanning defect image...")
    heatmap, flagged_windows, bad_lbp = scan_anomalies(
        bad_fabric, reference_hist, P, R, WINDOW_SIZE, STEP, threshold
    )
    
    # --- Visualize ---
    print("\n[6/6] Generating visualization...")
    visualize_results(
        good_fabric, bad_fabric, bad_lbp, heatmap,
        reference_hist, flagged_windows, threshold,
        WINDOW_SIZE, STEP, OUTPUT_HEATMAP
    )
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Good fabric shape: {good_fabric.shape}")
    print(f"Bad fabric shape:  {bad_fabric.shape}")
    print(f"Windows scanned:   {heatmap.shape[0] * heatmap.shape[1]}")
    print(f"Windows flagged:   {len(flagged_windows)}")
    print(f"Threshold used:    {threshold:.6f}")
    
    if len(flagged_windows) > 0:
        print("\nFlagged window details:")
        for i, w in enumerate(flagged_windows[:5]):  # Show first 5
            print(f"  Window {i+1}: position=({w['x']},{w['y']}), distance={w['distance']:.6f}")
        if len(flagged_windows) > 5:
            print(f"  ... and {len(flagged_windows) - 5} more")
    else:
        print("\n⚠️  No anomalies detected. Possible issues:")
        print("  - Threshold too high. Try lowering sigma_multiplier.")
        print("  - Defect not present in this image.")
        print("  - Window size or LBP parameters need tuning.")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
