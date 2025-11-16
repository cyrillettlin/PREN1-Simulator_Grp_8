import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Class can be much smaller without the demo methods and optional same/any.
class Rotation:
    def __init__(self, line1, line2, parallel_mode = "same"):
        self.p1a = np.asarray(line1[0], dtype=float)
        self.p1b = np.asarray(line1[1], dtype=float)
        self.p2a = np.asarray(line2[0], dtype=float)
        self.p2b = np.asarray(line2[1], dtype=float)
        if parallel_mode not in {"any", "same"}:
            raise ValueError("parallel_mode must be 'any' or 'same'")
        self.parallel_mode = parallel_mode
        # Define two vectors
        v1 = self.p1b - self.p1a
        v2 = self.p2b - self.p2a
        #Calculate the lenth of the vectors
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            raise ValueError("Lines must have non-zero length.")
        #Make vectors the same lenght.
        self.t1 = v1 / n1
        self.t2 = v2 / n2

        # Precompute the required rotation
        self._rotation_rad = self._compute_required_rotation()

        # Calculates the amount of rotation needed for the Vecors to point in the same direction. 
        # Signed angle u->v (in (-pi, pi])
    def signed_angle(self, u, v):
        """Return signed angle (radians) to rotate u → v, range (-π, π]."""
        cross_z = u[0]*v[1] - u[1]*v[0]   # equivalent to determinant |u v|
        dot     = u[0]*v[0] + u[1]*v[1]
        return np.arctan2(cross_z, dot)
        

    def _compute_required_rotation(self):
        """Compute how much line1 must rotate to align with line2."""
        delta_same = self.signed_angle(self.t1, self.t2)

        if self.parallel_mode == "same":
            return delta_same

        # Otherwise pick smaller rotation that yields parallelism
        delta_opposite = self.signed_angle(self.t1, -self.t2)
        return delta_same if abs(delta_same) <= abs(delta_opposite) else delta_opposite
    

    # Rotate a Shape function.
    def rotate_contour(self, contour, angle_deg, center_xy=None, keep_shape=True):
        """
        Rotate a single contour by angle_deg around its centroid or a given center.
        - contour: np.ndarray shaped (N, 2) or (N, 1, 2)
        - center_xy: optional (x, y). If None, uses contour centroid via cv.moments,
          falling back to arithmetic mean if m00 == 0 (degenerate area).
        - keep_shape: if True, return with the same shape as input.
        """
        if contour is None or len(contour) == 0:
            raise ValueError("rotate_contour: empty contour")

        # Remember original shape to restore later if requested
        orig_shape = contour.shape

        # Normalize to (N, 2), float
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)

    # TODO: Replace with the getCenterPoint Method of Puzzles!! Determine rotation center----------------------
        if center_xy is not None:
            cx, cy = center_xy
        else:
            M = cv.moments(pts)
            if abs(M["m00"]) > 1e-9:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                # Degenerate contour (area ~ 0): use arithmetic mean as fallback
                cx, cy = pts.mean(axis=0)
    # ----------------------------------------------------------------------------------------------------------
        center = np.array([cx, cy], dtype=np.float32)

        # Rotation matrix (2x2)
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_mat = np.array([[c, -s],
                            [s,  c]], dtype=np.float32)

        # translate → rotate → translate back
        rotated = (pts - center) @ rot_mat.T + center

        if keep_shape:
            # Restore original shape and dtype if you want
            rotated = rotated.reshape(orig_shape).astype(contour.dtype, copy=False)

        return rotated
    


    def show(self, contours, cmap='tab10', linewidth=2, title='Detected Contours'):
        """
        Displays all contours in a matplotlib window with colors, area info, and proper scaling.
        """
        if not contours:
            print("No contours to display.")
            return

        # Create a new figure
        plt.figure(figsize=(8, 8))
        colors = plt.get_cmap(cmap).colors

        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            #print(f"Contour {i}: Area = {area:.2f}")
            
            cnt = np.asarray(cnt).reshape(-1, 2)
            plt.plot(cnt[:, 0], cnt[:, 1],
                    color=colors[i % len(colors)],
                    linewidth=linewidth,
                    label=f'Contour {i} (Area={area:.1f})')

        # Improve visualization
        plt.title(title)
        plt.gca().invert_yaxis()       # Match OpenCV image coordinates
        plt.axis("equal")              # Keep aspect ratio correct
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(fontsize='small', loc='upper right', ncol=2)
        plt.tight_layout()
        
        # Display window
        plt.show()



    # ---------- public getters ----------
    def rotation_rad(self):
        """Signed rotation (radians) to make line1 parallel to line2."""
        return self._rotation_rad

    def rotation_deg(self):
        """Signed rotation (degrees) to make line1 parallel to line2."""
        return np.rad2deg(self._rotation_rad)
    
if __name__ == "__main__":
    from edgedetection import EdgeDetection

    # Rotation Demo/ Walktrough. This Demo Allgns Puzzle 1 with Puzzle 4. 
    ##Get Puzzles using EdgeDetection
    path = "Data/best_example.jpg"
    detector = EdgeDetection(path)
    detector.load()
    detector.find_contours()
    detector.filter_contours()

    puzzles = detector.get_puzzle_pieces()
    puzzle1 = puzzles[0]
    puzzle3 = puzzles[2]
    puzzle4 = puzzles[3]



    line1 = puzzle1.get_best_4_corners()[0], puzzle1.get_best_4_corners()[1]
    line2 = puzzle4.get_best_4_corners()[1], puzzle3.get_best_4_corners()[0]

    #print(puzzle1.get_best_4_corners()[0], puzzle1.get_best_4_corners()[1])

    aligner= Rotation(line1, line2, parallel_mode="same")
    angle = aligner.rotation_deg()
    print(angle)

    #--------------------Visualization--------------------

    ## Get contours
    contours = [p.contour for p in puzzles]
    center = puzzle1.get_center_point()
    ## Add my rotated contour
    cnt5 = aligner.rotate_contour(contours[0],angle, center) 
    contours.append(cnt5)

    detector.show_result()
    aligner.show(contours)



