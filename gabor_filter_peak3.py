import cv2
import numpy as np

# 1. Load image and apply local contrast normalization
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: 1.jpg not found.")
    exit()

blur = cv2.GaussianBlur(img, (101, 101), 0)
normalized_img = cv2.divide(img, blur, scale=255)

# 2. Updated parameters extracted from the backlighting FFT
theta = 1.101          # Radians (63.08 degrees)
lambd = 6.25           # Wavelength in pixels
ksize = 19             # Kernel size (approx 3 * lambda)
sigma = 0.56 * lambd   # Gaussian envelope sigma
gamma = 0.5            # Spatial aspect ratio
psi = 0 

# 3. Compute DC-balanced Quadrature Gabor Kernels
kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)

# Subtract mean to eliminate residual direct current offset response
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

# 6. Evaluate fabric uniformity
anomalous_mask = (energy_map < lower_bound) | (energy_map > upper_bound)
anomalous_pixels = np.sum(anomalous_mask)
total_pixels = energy_map.size
defect_ratio = (anomalous_pixels / total_pixels) * 100
status = "PASS" if defect_ratio <= 0.05 else "FAIL"

# Save binary mask output
cv2.imwrite('defects.png', (anomalous_mask * 255).astype(np.uint8))

print("=" * 45)
print("   CALIBRATED FABRIC STRUCTURAL TEST   ")
print("=" * 45)
print(f"Mean Energy (μ)         : {mu:.4f}")
print(f"Standard Deviation (σ)  : {sigma_stat:.4f}")
print(f"Valid Energy Range      : [{lower_bound:.4f} to {upper_bound:.4f}]")
print(f"Anomalous Pixels Found  : {anomalous_pixels} out of {total_pixels}")
print(f"Defect Surface Ratio    : {defect_ratio:.4f}%")
print("-" * 45)
print(f"FINAL INSPECTION STATUS : {status}")
print("=" * 45)
cv2.imwrite('defects.png', ((energy_map < lower_bound) | (energy_map > upper_bound) * 255).astype(np.uint8))