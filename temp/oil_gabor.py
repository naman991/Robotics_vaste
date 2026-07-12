import cv2
import numpy as np

# 1. Load image and normalize illumination
img = cv2.imread('d1.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: d1.jpg not found.")
    exit()

blur = cv2.GaussianBlur(img, (101, 101), 0)
normalized_img = cv2.divide(img, blur, scale=255)

# 2. Structural Gabor Parameters (From your latest FFT)
theta = -1.141
lambd = 6.50
ksize = 19
sigma = 0.56 * lambd
gamma = 0.5

# 3. Compute Structural Energy Map
kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
kernel_real -= kernel_real.mean()
kernel_imag -= kernel_imag.mean()

f_real = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_real)
f_imag = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_imag)
structural_energy = np.sqrt(f_real**2 + f_imag**2)

# 4. CHANNEL 1: Structural Analysis (Tears, Cuts, Double Picks)
mu_struct = np.mean(structural_energy)
sigma_struct = np.std(structural_energy)
struct_mask = (structural_energy < (mu_struct - 3*sigma_struct)) | (structural_energy > (mu_struct + 3*sigma_struct))

# 5. CHANNEL 2: Intensity Analysis (Oil Stains, Blemishes)
mu_intensity = np.mean(normalized_img)
sigma_intensity = np.std(normalized_img)
# Oil stains appear darker under backlighting; flag pixels dropping below 3.5 standard deviations
intensity_mask = normalized_img < (mu_intensity - 3.0 * sigma_intensity)

# 6. Combine Channels & Filter Noise
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
cv2.imwrite('defect_oil.png', final_macro_mask)

print("=" * 45)
print(" DUAL-CHANNEL FABRIC INSPECTION REPORT ")
print("=" * 45)
print(f"Structural Outliers     : {np.sum(struct_mask)}")
print(f"Intensity/Stain Outliers: {np.sum(intensity_mask)}")
print("-" * 45)
print(f"Continuous Macro Blocks : {defect_count}")
print(f"Largest Defect Block    : {max_size} px")
print(f"FINAL INSPECTION STATUS : {status}")
print("=" * 45)