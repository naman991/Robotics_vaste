import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image in grayscale
IMAGE_PATH = 'd1.jpg'
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: {IMAGE_PATH} not found.")
    exit()

h, w = img.shape

# 1. Extract a Small Central 256x256 Patch for Lean Calibration
P_SIZE = 256
cx_img, cy_img = w // 2, h // 2
patch = img[cy_img - P_SIZE//2 : cy_img + P_SIZE//2, cx_img - P_SIZE//2 : cx_img + P_SIZE//2]

# 2. Computationally Lean Parameter Extraction via 2D RFFT
# rfft2 operates only on real numbers, cutting execution memory and time in half.
f_transform = np.fft.rfft2(patch)
magnitude_spectrum = np.abs(f_transform)

# Mask the DC component in the unshifted RFFT plane
# In rfft2, DC (0-frequency) sits at [0,0]. Low frequencies sit at the corners of the first column.
magnitude_spectrum[:15, :15] = 0
magnitude_spectrum[-14:, :15] = 0

# Find the dominant peak in the halved rfft2 spectrum
flat_indices = np.argsort(magnitude_spectrum.flatten())[::-1]
patch_h, patch_w = magnitude_spectrum.shape  # 256 x 129

fx, fy = 0, 0
for idx in flat_indices:
    y_idx, x_idx = divmod(idx, patch_w)
    
    # Map RFFT indices back to spatial frequency cycles per patch
    curr_fx = x_idx
    curr_fy = y_idx if y_idx <= P_SIZE // 2 else y_idx - P_SIZE
    
    if curr_fx == 0 and curr_fy == 0:
        continue
        
    fx = curr_fx * (w / P_SIZE)
    fy = curr_fy * (h / P_SIZE)
    break

# Save the diagnostic FFT plot to disk instead of pausing execution
vis_spectrum = np.log(magnitude_spectrum + 1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(patch, cmap='gray'), plt.title('256x256 Calibration Patch')
plt.subplot(1, 2, 2), plt.imshow(vis_spectrum, cmap='jet'), plt.title('Lean RFFT Spectrum')
plt.colorbar()
plt.savefig('fft_spectrum_diagnostic.png', bbox_inches='tight')
plt.close()

# 3. Dynamic Gabor Parameter Calculation
theta = np.arctan2(fy, fx)
# Calculate pixel wavelength relative to the physical patch size
lambd = P_SIZE / np.sqrt(fx**2 + fy**2)
ksize = int(3 * lambd)
if ksize % 2 == 0:
    ksize += 1

sigma = 0.56 * lambd
gamma = 0.5

# #DIAGNOSIS
# # =====================================================================
# # DIAGNOSTIC TEST BLOCK: PARAMETER ISOLATION
# # =====================================================================
# print("\n" + "="*50)
# print(" DIAGNOSTIC ANALYSIS: GABOR & SPECTRAL ALIGNMENT")
# print("="*50)
# print(f"Target Ground Truth (Script 1) : fx=187, fy=-173, lambda=7.85px, theta=-0.7465 rad")
# print(f"Current Script 2 Spectrum Peak : fx={fx:.4f}, fy={fy:.4f}")
# print(f"Current Script 2 Wavelength    : lambd={lambd:.4f} px")
# print(f"Current Script 2 Kernel Size   : ksize={ksize}x{ksize}")
# print(f"Current Script 2 Theta (Angle) : theta={theta:.4f} rad ({np.degrees(theta):.2f}°)")

# # Check mathematical correctness of wavelength scale factor
# expected_global_lambd = w / np.sqrt(fx**2 + fy**2) if (fx**2 + fy**2) > 0 else 0
# print(f"Recalculated Global Wavelength : {expected_global_lambd:.4f} px")

# # Verify filter geometry distortion
# if ksize <= 3:
#     print("CRITICAL ALERT: Gabor kernel has collapsed to a 3x3 noise filter.")
# if np.sign(theta) != np.sign(-0.7465):
#     print("CRITICAL ALERT: Theta angle phase inversion detected (Conjugate peak mismatch).")
# print("="*50 + "\n")
# exit()
# # =====================================================================
# #DIAGNOSIS

# 4. Full-Resolution Illumination Normalization // removed gaussian blur
img_mean = np.mean(img)
normalized_img = cv2.multiply(img, 1.0 / img_mean, scale=128, dtype=cv2.CV_8U)

# 5. Full-Resolution Structural Energy Map Computation
kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
kernel_real -= kernel_real.mean()
kernel_imag -= kernel_imag.mean()

f_real = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_real)
f_imag = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_imag)
structural_energy = np.sqrt(f_real**2 + f_imag**2)

# 6. CHANNEL 1: Structural Analysis
mu_struct = np.mean(structural_energy)
sigma_struct = np.std(structural_energy)
struct_mask = (structural_energy < (mu_struct - 3 * sigma_struct)) | (structural_energy > (mu_struct + 3 * sigma_struct))

# 7. CHANNEL 2: Intensity Analysis
mu_intensity = np.mean(normalized_img)
sigma_intensity = np.std(normalized_img)
intensity_mask = normalized_img < (mu_intensity - 3.0 * sigma_intensity)

# 8. Combine Channels & Macro-Filtering
combined_pixel_mask = (struct_mask | intensity_mask).astype(np.uint8) * 255
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_pixel_mask)

MIN_CLUSTER_SIZE = 500
final_macro_mask = np.zeros_like(combined_pixel_mask)
defect_count = 0
max_size = 0

for i in range(1, num_labels):
    cluster_size = stats[i, cv2.CC_STAT_AREA]
    if cluster_size >= MIN_CLUSTER_SIZE:
        final_macro_mask[labels == i] = 255
        defect_count += 1
        if cluster_size > max_size:
            max_size = cluster_size

status = "FAIL" if defect_count > 0 else "PASS"
cv2.imwrite('defect.png', final_macro_mask)

print("=" * 45)
print(" DUAL-CHANNEL FABRIC INSPECTION REPORT ")
print("=" * 45)
print(f"Computed Patch Peaks    : fx={fx}, fy={fy}")
print(f"Calibrated Theta        : {theta:.4f} rad")
print(f"Calibrated Lambda       : {lambd:.4f} px (ksize={ksize})")
print("-" * 45)
print(f"Structural Outliers     : {np.sum(struct_mask)} px")
print(f"Intensity/Stain Outliers: {np.sum(intensity_mask)} px")
print("-" * 45)
print(f"Continuous Macro Blocks : {defect_count}")
print(f"Largest Defect Block    : {max_size} px")
print(f"FINAL INSPECTION STATUS : {status}")
print("=" * 45)