import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2 as cv

# Waiting for GLobalArea to measure distances.
class Translation:

    def __init__(self):
        pass
        

    def translate_puzzle_in_place(self, puzzle, trans):

        dx_f, dy_f  = trans

        cnt = puzzle.get_contour() if hasattr(puzzle, "get_contour") else puzzle.contour
        arr = np.asarray(cnt)
        orig_shape = arr.shape

        pts = arr.reshape(-1, 2).astype(np.float32, copy=False)
        pts[:, 0] += dx_f
        pts[:, 1] += dy_f

        new_cnt = pts.reshape(orig_shape).astype(np.float32, copy=False)
        if hasattr(puzzle, "set_contour"):
            puzzle.set_contour(new_cnt)
        else:
            puzzle.contour = new_cnt

        if hasattr(puzzle, "corners") and puzzle.corners is not None:
            c = np.asarray(puzzle.corners, dtype=np.float32).reshape(-1, 2)
            c[:, 0] += dx_f
            c[:, 1] += dy_f
            puzzle.corners = c.reshape(4, 2).astype(np.float32, copy=False)

        return puzzle
    
    def delta_xy(self, p1, p2):
        """
        Return the translation needed to move p1 onto p2.

        p1, p2: array-like (x, y)
        returns: (dx, dy) where p1 + (dx,dy) = p2
        """
        a = np.asarray(p1, dtype=np.float32).reshape(2,)
        b = np.asarray(p2, dtype=np.float32).reshape(2,)
        dx, dy = (b - a)
        return float(dx), float(dy)
        
    # Convenience

    def translate_piece_b_to_a_in_place(self, puzzle_b, lines, use_midpoint: bool = True):
        """
        Translate puzzle_b IN PLACE so its matched edge aligns to the other edge.

        lines: (lineA, lineB) from ga.get_matching_edge_lines(...)
               where lineA = (a0, a1) and lineB = (b0, b1)

        By default aligns midpoints (robust). If use_midpoint=False, aligns first endpoint with second endpoint.
        Returns (dx, dy) applied.
        """
        (a0, a1), (b0, b1) = lines
        a0 = np.asarray(a0, dtype=np.float32); a1 = np.asarray(a1, dtype=np.float32)
        b0 = np.asarray(b0, dtype=np.float32); b1 = np.asarray(b1, dtype=np.float32)

        if use_midpoint:
            a_ref = 0.5 * (a0 + a1)
            b_ref = 0.5 * (b0 + b1)
        else:
            a_ref = a0
            b_ref = b1

        dx, dy = (a_ref - b_ref)
        dx_f, dy_f = float(dx), float(dy)

        # --- contour (preserve shape) ---
        cnt = puzzle_b.get_contour() if hasattr(puzzle_b, "get_contour") else puzzle_b.contour
        arr = np.asarray(cnt)
        orig_shape = arr.shape

        pts = arr.reshape(-1, 2).astype(np.float32, copy=False)
        pts[:, 0] += dx_f
        pts[:, 1] += dy_f

        new_cnt = pts.reshape(orig_shape).astype(np.float32, copy=False)
        if hasattr(puzzle_b, "set_contour"):
            puzzle_b.set_contour(new_cnt)
        else:
            puzzle_b.contour = new_cnt

        # --- corners ---
        if hasattr(puzzle_b, "corners") and puzzle_b.corners is not None:
            c = np.asarray(puzzle_b.corners, dtype=np.float32).reshape(-1, 2)
            c[:, 0] += dx_f
            c[:, 1] += dy_f
            puzzle_b.corners = c.reshape(4, 2).astype(np.float32, copy=False)

        return dx_f, dy_f
