import cv2
import numpy as np

# 1. Pipeline Re-execution to get raw energy map and mask
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: 1.jpg not found.")
    exit()

blur = cv2.GaussianBlur(img, (101, 101), 0)
normalized_img = cv2.divide(img, blur, scale=255)

theta, lambd, ksize = 1.101, 6.25, 19
sigma = 0.56 * lambd
gamma = 0.5

kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
kernel_real -= kernel_real.mean()
kernel_imag -= kernel_imag.mean()

f_real = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_real)
f_imag = cv2.filter2D(normalized_img, cv2.CV_64F, kernel_imag)
energy_map = np.sqrt(f_real**2 + f_imag**2)

mu = np.mean(energy_map)
sigma_stat = np.std(energy_map)
lower_bound = mu - (3 * sigma_stat)
upper_bound = mu + (3 * sigma_stat)

# Create boolean masks for the two zones
anomaly_mask = (energy_map < lower_bound) | (energy_map > upper_bound)
background_mask = ~anomaly_mask

# 2. Extract Zone-Specific Statistics
energy_anomalies = energy_map[anomaly_mask]
energy_background = energy_map[background_mask]

# 3. Structural Continuity Check (Connected Components Analysis)
binary_mask_uint8 = (anomaly_mask * 255).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_uint8)
# stats standard output: [left, top, width, height, total_area_in_pixels]
sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude background component

print("=" * 50)
print("        DIAGNOSTIC SOURCE-OF-ERROR REPORT        ")
print("=" * 50)
print(f"BACKGROUND ZONE (PASSING FABRIC):")
print(f"  Mean Energy                  : {np.mean(energy_background):.4f}")
print(f"  Energy Variance              : {np.var(energy_background):.4f}")
print("-" * 50)
print(f"ANOMALY ZONE (FAILING PIXELS):")
print(f"  Mean Energy                  : {np.mean(energy_anomalies):.4f}")
print(f"  Energy Variance              : {np.var(energy_anomalies):.4f}")
print("-" * 50)
print(f"GEOMETRIC MAPPING EVIDENCE:")
print(f"  Total Discrete Micro-Objects : {len(sizes)}")
print(f"  Average Object Size (Pixels) : {np.mean(sizes):.2f} px")
print(f"  Largest Single Object Size   : {np.max(sizes) if len(sizes) > 0 else 0} px")
print("=" * 50)