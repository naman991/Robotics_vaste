import cv2
import numpy as np

# 1. Load the defect image and apply local contrast normalization
img = cv2.imread('d2.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: d2.jpg not found.")
    exit()

blur = cv2.GaussianBlur(img, (101, 101), 0)
normalized_img = cv2.divide(img, blur, scale=255)

# 2. Defect image parameters extracted from the new FFT
theta = -1.141         # Radians (-42.78 degrees)
lambd = 6.50           # Wavelength in pixels
ksize = 19             # Kernel size (3 * lambda)
sigma = 0.56 * lambd   # Gaussian envelope sigma
gamma = 0.5            # Elongated aspect ratio
psi = 0 

# 3. Compute DC-balanced Quadrature Gabor Kernels
kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)

kernel_real -= kernel_real.mean()
kernel_imag -= kernel_imag.mean()

# 4. Generate magnitude energy map
f_real = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_real)
f_imag = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_imag)
energy_map = np.sqrt(f_real**2 + f_imag**2)

# 5. Statistical Baseline Calculation
mu = np.mean(energy_map)
sigma_stat = np.std(energy_map)

lower_bound = mu - (3 * sigma_stat)
upper_bound = mu + (3 * sigma_stat)

# 6. High-Resolution Pixel Thresholding
pixel_mask = (energy_map < lower_bound) | (energy_map > upper_bound)
binary_mask_uint8 = (pixel_mask * 255).astype(np.uint8)

# 7. Structural Macro-Filtering (Connected Components)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_uint8)

# Industrial Constraints: Drop noise clusters under 500 contiguous pixels
MIN_CLUSTER_SIZE = 500
macro_mask = np.zeros_like(binary_mask_uint8)
large_defect_count = 0
max_defect_size = 0

for i in range(1, num_labels):  # Index 0 is background
    cluster_size = stats[i, cv2.CC_STAT_AREA]
    if cluster_size >= MIN_CLUSTER_SIZE:
        macro_mask[labels == i] = 255
        large_defect_count += 1
        if cluster_size > max_defect_size:
            max_defect_size = cluster_size

# Final Evaluation: System triggers FAIL only if continuous macro defects exist
status = "FAIL" if large_defect_count > 0 else "PASS"

# Save the structurally filtered defect map
cv2.imwrite('defects.png', macro_mask)

print("=" * 45)
print("   MACRO-STRUCTURAL FABRIC INSPECTION  ")
print("=" * 45)
print(f"Mean Energy (μ)         : {mu:.4f}")
print(f"Standard Deviation (σ)  : {sigma_stat:.4f}")
print(f"Total Raw Pixel Outliers: {np.sum(pixel_mask)}")
print("-" * 45)
print(f"Continuous Defect Blocks: {large_defect_count}")
print(f"Largest Defect Block    : {max_defect_size} px")
print(f"FINAL INSPECTION STATUS : {status}")
print("=" * 45)