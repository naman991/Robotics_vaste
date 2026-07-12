import cv2
import numpy as np

# 1. Load image in grayscale
IMAGE_PATH = 'd1.jpg'
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: {IMAGE_PATH} not found.")
    exit()

h, w = img.shape

# 2. Extract Dominant Structural Frequency via 2D FFT
f_transform = np.fft.fft2(img)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = np.abs(f_shift)

# Mask the DC component (center) to avoid 0-frequency peak
cx, cy = w // 2, h // 2
magnitude_spectrum[cy-5:cy+6, cx-5:cx+6] = 0

flat_indices = np.argsort(magnitude_spectrum.flatten())[::-1]

fx, fy = 0, 0
detected = []
for idx in flat_indices:
    y_idx, x_idx = divmod(idx, w)
    curr_fx = x_idx - cx
    curr_fy = cy - y_idx
    
    if any(np.allclose([curr_fx, curr_fy], [d[0], d[1]], atol=3) or np.allclose([-curr_fx, -curr_fy], [d[0], d[1]], atol=3) for d in detected):
        continue
        
    fx, fy = curr_fx, curr_fy
    break

# 3. Dynamic Gabor Parameter Calculation
theta = np.arctan2(fy, fx)
lambd = w / np.sqrt(fx**2 + fy**2)
ksize = int(3 * lambd)
if ksize % 2 == 0:
    ksize += 1

sigma = 0.56 * lambd
gamma = 0.5

# 4. Illumination Normalization
blur = cv2.GaussianBlur(img, (101, 101), 0)
normalized_img = cv2.divide(img, blur, scale=255)

# 5. Compute Structural Energy Map
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
print(f"Computed Peak Parameters: fx={fx}, fy={fy}")
print(f"Calibrated Theta        : {theta:.4f} rad")
print(f"Calibrated Lambda       : {lambd:.4f} px (ksize={ksize})")
print("-" * 45)
print(f"Structural Outliers     : {np.sum(struct_mask)}")
print(f"Intensity/Stain Outliers: {np.sum(intensity_mask)}")
print("-" * 45)
print(f"Continuous Macro Blocks : {defect_count}")
print(f"Largest Defect Block    : {max_size} px")
print(f"FINAL INSPECTION STATUS : {status}")
print("=" * 45)