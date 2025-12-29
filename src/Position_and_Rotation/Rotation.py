import numpy as np
import math 
import cv2 as cv
from collections import deque
from typing import Any, Dict, List, Set

class Rotation:
    """
    Rotation utilities.
    No parameters required at instantiation.
    Lines and contours are passed directly to methods.
    """

    def __init__(self):
        pass

    # ---------- helpers ----------
    def _unit_direction(self, obj):
        """
        Return unit direction vector (2,) from either:
        - a line given by 2 endpoints: shape (2,2) or (2,1,2)
        - a direction vector: shape (2,)
        """
        a = np.asarray(obj, dtype=np.float64)

        # Case 1: direction vector (2,)
        if a.shape == (2,):
            v = a

        # Case 2: endpoints (2,2)
        elif a.shape == (2, 2):
            v = a[1] - a[0]

        # Case 3: OpenCV style endpoints (2,1,2)
        elif a.shape == (2, 1, 2):
            v = a[1, 0] - a[0, 0]

        else:
            raise ValueError(f"_unit_direction: expected (2,), (2,2), or (2,1,2), got {a.shape}")

        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("_unit_direction: zero-length direction")
        return v / n


    def signed_angle(self, u, v):
        """Return signed angle (radians) to rotate u → v, range (-π, π]."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cross_z = u[0] * v[1] - u[1] * v[0]
        dot = u[0] * v[0] + u[1] * v[1]
        return np.arctan2(cross_z, dot)

    # ---------- rotation between lines ----------
    def compute_required_rotation_rad(self, line1, line2, parallel_mode="opposite"):
        """
        Compute signed rotation (radians) needed to rotate line1 same
        so it becomes parallel to line2.
        """
        if parallel_mode not in {"any", "same", "opposite"}:
            raise ValueError("parallel_mode must be 'any', 'same', or 'opposite'")

        t1 = self._unit_direction(line1)
        t2 = self._unit_direction(line2)

        delta_same = self.signed_angle(t1, t2)
        delta_opposite = self.signed_angle(t1, -t2)

        if parallel_mode == "same":
            return delta_same
        if parallel_mode == "opposite":
            return delta_opposite
        #Return smallest angle
        return delta_same if abs(delta_same) <= abs(delta_opposite) else delta_opposite

    def compute_required_rotation_deg(self, lines, parallel_mode="opposite"):
        """
        Computes rotation for piece 2.
        lines: (line1, line2) where each line is (p0, p1)
        """
        line1, line2 = lines
        return np.rad2deg(
            self.compute_required_rotation_rad(line2, line1, parallel_mode)
        )

    # ---------- contour rotation ----------
    def rotate_contour(self, contour, angle_deg, center_xy=None, keep_shape=True):
        if contour is None or len(contour) == 0:
            raise ValueError("rotate_contour: empty contour")

        orig = np.asarray(contour)
        orig_shape = orig.shape
        pts = orig.reshape(-1, 2).astype(np.float32)

        # Determine rotation center
        if center_xy is not None:
            cx, cy = center_xy
        else:
            M = cv.moments(pts)
            if abs(M["m00"]) > 1e-9:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = pts.mean(axis=0)

        center = np.array([cx, cy], dtype=np.float32)

        angle_rad = np.deg2rad(angle_deg).astype(np.float32)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_mat = np.array([[c, -s],
                            [s,  c]], dtype=np.float32)

        rotated = (pts - center) @ rot_mat.T + center

        if keep_shape:
            rotated = rotated.reshape(orig_shape).astype(np.float32, copy=False)

        return rotated

    def rotate_puzzle_in_place(self, puzzle, angle_deg, center_xy=None):
        cnt = puzzle.get_contour() if hasattr(puzzle, "get_contour") else puzzle.contour
        pts = np.asarray(cnt, dtype=np.float32).reshape(-1, 2)

        # compute center once
        if center_xy is None:
            M = cv.moments(pts)
            if abs(M["m00"]) > 1e-9:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = pts.mean(axis=0)
            center_xy = (float(cx), float(cy))

        # rotate contour using that center
        cnt_rot = self.rotate_contour(cnt, angle_deg, center_xy=center_xy, keep_shape=True)

        if hasattr(puzzle, "set_contour"):
            puzzle.set_contour(cnt_rot)
        else:
            puzzle.contour = cnt_rot

        # rotate corners using the SAME center
        if hasattr(puzzle, "corners") and puzzle.corners is not None:
            crn = np.asarray(puzzle.corners, dtype=np.float32).reshape(-1, 2)

            cx, cy = center_xy
            center = np.array([cx, cy], dtype=np.float32)

            angle_rad = np.deg2rad(angle_deg).astype(np.float32)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            rot_mat = np.array([[c, -s],
                                [s,  c]], dtype=np.float32)

            crn_rot = (crn - center) @ rot_mat.T + center
            puzzle.corners = crn_rot.reshape(4, 2).astype(np.float32)

        return puzzle



    def _unit(self, v):
        v = np.asarray(v, dtype=np.float32).reshape(2,)
        n = np.linalg.norm(v)
        return v / (n + 1e-9)

    def _signed_angle_rad(self, v_from, v_to) -> float:
        """Signed angle (rad) rotating v_from -> v_to."""
        a = self._unit(v_from)
        b = self._unit(v_to)
        cross = a[0]*b[1] - a[1]*b[0]
        dot   = a[0]*b[0] + a[1]*b[1]
        return float(np.arctan2(cross, dot))

    def anchor_rotation_for_corner_deg(self, piece, flat_edges, dir1, dir2, tol_dot=0.95):
        """
        Returns angle_deg to rotate this piece so that the two flat edges
        from their shared corner point align with the given directions.
        dir = (x,y)
        Right (-1,0)
        Left (1,0)
        Down (0,1)
        Up (0,-1)


        flat_edges: two edge indices (0..3)
        Assumes piece.corners are in order [TL, TR, BR, BL] clockwise.
        Edge convention: 0:(0->1), 1:(1->2), 2:(2->3), 3:(3->0)
        """
        EDGE_TO = ((0, 1), (1, 2), (2, 3), (3, 0))

        corners = np.asarray(piece.corners, dtype=np.float32).reshape(4, 2)
        e0, e1 = int(flat_edges[0]) % 4, int(flat_edges[1]) % 4

        # --- find common corner index of the two flat edges ---
        s0 = set(EDGE_TO[e0])
        s1 = set(EDGE_TO[e1])
        common = list(s0 & s1)
        if len(common) != 1:
            raise ValueError(f"flat edges {flat_edges} do not share exactly one corner")
        k = common[0]  # common corner index 0..3

        def vec_from_common(edge_idx):
            i, j = EDGE_TO[edge_idx]
            other = j if i == k else i  # the corner on that edge that's not the common corner
            return corners[other] - corners[k]  # vector starts at common corner

        v0 = self._unit(vec_from_common(e0))
        v1 = self._unit(vec_from_common(e1))

        TOP   = self._unit(np.array(dir1, dtype=np.float32))
        RIGHT = self._unit(np.array(dir2, dtype=np.float32))

        def rotate_vec(v, ang_rad):
            c, s = np.cos(ang_rad), np.sin(ang_rad)
            R = np.array([[c, -s],
                          [s,  c]], dtype=np.float32)
            return self._unit(R @ v)

        # Try two assignments:
        # A) v0 -> TOP, v1 -> RIGHT
        # B) v1 -> TOP, v0 -> RIGHT
        candidates = []
        for vh, vv in [(v0, v1), (v1, v0)]:
            ang = self._signed_angle_rad(vh, TOP)     # rotate this piece by ang
            vv_rot = rotate_vec(vv, ang)

            score = float(np.dot(vv_rot, RIGHT))      # want +1 (same direction), not -1
            ok = score >= tol_dot
            candidates.append((ok, score, ang))

        # pick best: first any "ok", otherwise best score
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best_ang = candidates[0][2]
        return float(np.rad2deg(best_ang))
