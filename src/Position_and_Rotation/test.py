import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Create some fake OpenCV-style contours in pixel coordinates
contour1 = np.array([[[10, 10]], [[60, 10]], [[60, 40]], [[10, 40]]], dtype=np.int32)  # square
contour2 = np.array([[[100, 50]], [[120, 80]], [[80, 80]]], dtype=np.int32)            # triangle
contour3 = np.array([[[200, 200]], [[230, 200]], [[230, 230]], [[200, 230]]], dtype=np.int32)

contours_px = [contour1, contour2, contour3]

# --- Step 2: Define scale (pixel → mm) and offset (origin shift)
pixels_to_mm_ratio = (0.25, 0.25)  # each pixel = 0.25 mm
offset_mm = (0.0, 0.0)             # no offset (same origin)

# --- Step 3: Apply uniform scaling to the entire group
scaled_contours = []
for cnt in contours_px:
    cnt_float = cnt.reshape(-1, 2).astype(float)
    cnt_scaled = cnt_float * [pixels_to_mm_ratio[0], pixels_to_mm_ratio[1]] + offset_mm
    scaled_contours.append(cnt_scaled)

# --- Step 4: Visualize before & after
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].set_title("Original contours (pixels)")
axes[1].set_title("Scaled contours (millimeters)")

def plot_contours(ax, contours, color):
    for cnt in contours:
        c = np.vstack([cnt, cnt[0]])  # close the loop
        ax.plot(c[:,0], c[:,1], color=color, lw=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

plot_contours(axes[0], [c.reshape(-1,2) for c in contours_px], 'tab:blue')
plot_contours(axes[1], scaled_contours, 'tab:green')

for ax in axes:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.tight_layout()
plt.show()

# --- Step 5: Inspect numerically
print("Original (pixels):", contour1.reshape(-1,2))
print("Scaled (mm):", scaled_contours[0])
