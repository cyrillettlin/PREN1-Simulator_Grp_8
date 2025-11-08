import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2 as cv

class PositionAndRotation:

    def __init__(self, puzzle):
        self.contour

    # Translate a contour.
    def translate_contour(contour, x_axis, y_axis):
        """
        Move a contour by x and y coordinates.
        """
        return contour + np.array([x_axis, y_axis])

# Rotate a Shape function.
    def rotate_contour(contour, angle_deg):
        M = cv.moments(contour)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center = np.array([cx, cy])
    
        angle_rad = np.deg2rad(angle_deg)
        rot_mat = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])

        # translate → rotate → translate back
        return (contour - center) @ rot_mat.T + center


    def plot_contours(contours, cmap='tab10', linewidth=2):
        colors = plt.get_cmap(cmap).colors
        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            print(f"Contour {i}: Area= {area}")
            cnt = np.asarray(cnt).reshape(-1, 2)
            plt.plot(cnt[:, 0], cnt[:, 1],
                    color=colors[i % len(colors)],
                    linewidth=linewidth)

    def plot_center(contours):
        for i, cnt in enumerate(contours):
            M = cv.moments(cnt)
            cx = M["m10"] / abs(M["m00"])
            cy = M["m01"] / abs(M["m00"])
            center = np.array([cx, cy])
            plt.scatter(center[0], center[1], color='red', marker='x', s=100)

    def draw_box(contours, ax=None, color='green', linewidth=2, text_offset=5):
        """
        Draw axis-aligned bounding boxes around contours and label them by index.

        Parameters:
            contours : list[np.ndarray]
                List of contours (each Nx2 or Nx1x2 array)
        """
        if ax is None:
            ax = plt.gca()

        for i, cnt in enumerate(contours):
            pts = np.asarray(cnt).reshape(-1, 2).astype(np.int32)
            x, y, w, h = cv.boundingRect(pts)

            # Compute and print area to console
            area = w * h
            print(f"Contour {i}: Bounding box area = {area}")
            # Draw rectangle
            rect = patches.Rectangle((x, y), w, h,
                                    fill=False, color=color, linewidth=linewidth)
            ax.add_patch(rect)
            # Label with contour index
            ax.text(x, y - text_offset, str(i),
                    color=color, fontsize=10, weight='bold')

    def plotAll():
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.autoscale(False)
        plt.show()
