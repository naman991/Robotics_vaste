import cv2
import numpy as np

# 1. Load image in grayscale
img = cv2.imread('d1.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: d1.jpg not found.")
    exit()

h, w = img.shape

# 2. Compute 2D FFT and shift DC component to center
f_transform = np.fft.fft2(img)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = np.abs(f_shift)

# 3. Mask the DC component (center) to avoid detecting the 0-frequency peak
cx, cy = w // 2, h // 2
magnitude_spectrum[cy-5:cy+6, cx-5:cx+6] = 0

# 4. Find the top 3 dominant peaks (excluding symmetry duplicates)
# Flatten and get sorted indices of highest intensities
flat_indices = np.argsort(magnitude_spectrum.flatten())[::-1]

print(f"Image Dimensions: Width={w}, Height={h}")
print("\nTop Frequency Peaks (Relative to Center 0,0):")
print(f"{'Peak':<5} | {'fx (Horizontal)':<15} | {'fy (Vertical)':<13}")
print("-" * 40)

detected = []
peak_count = 1

for idx in flat_indices:
    y_idx, x_idx = divmod(idx, w)
    fx = x_idx - cx
    fy = cy - y_idx  # Invert to match standard Cartesian coordinates
    
    # Avoid printing identical or symmetric peaks (-fx, -fy)
    if any(np.allclose([fx, fy], [d[0], d[1]], atol=3) or np.allclose([-fx, -fy], [d[0], d[1]], atol=3) for d in detected):
        continue
        
    print(f"#{peak_count:<4} | {fx:<15} | {fy:<13}")
    detected.append((fx, fy))
    peak_count += 1
    if peak_count > 3:
        break