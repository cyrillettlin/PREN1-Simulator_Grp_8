from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from puzzle import Puzzle


@dataclass(frozen=True)
class Rect:
    """
    Axis-aligned rectangle on a 2D physical plane (e.g., table surface).

    Coordinate system:
      • Origin (0,0) = chosen reference point on the table (e.g., top-left corner)
      • +X → right, +Y → forward or down (depending on your convention)
      • Units are millimeters (recommended)
    """
    x: float  # X coordinate of the rectangle’s top-left corner (mm)
    y: float  # Y coordinate of the rectangle’s top-left corner (mm)
    w: float  # width (mm)
    h: float  # height (mm)

    @property
    def right(self) -> float:
        """X coordinate of the right edge."""
        return self.x + self.w

    @property
    def bottom(self) -> float:
        """Y coordinate of the bottom edge."""
        return self.y + self.h


class GlobalArea:

    """
    Defines a global 2D coordinate plane (the work surface) with:
      - a base rectangle representing the total physical area
      - an UNSOLVED area (where the loose puzzle pieces start)
      - a SOLVED area (where the finished puzzle should be assembled)

    All positions are defined in absolute coordinates on that same plane.
    """
    def __init__(
        self,
        base: Optional["Rect"] = None,
        area_unsolved: Optional["Rect"] = None,
        area_solved: Optional["Rect"] = None,
        *,
        pixels_to_mm_ratio: Tuple[float, float] = (0.25, 0.25)
    ) -> None:
        """
        Parameters
        ----------
        base : Rect, optional
            The total working area (absolute position and size).
        area_unsolved : Rect, optional
            Exact coordinates of the unsolved zone within the plane.
        area_solved : Rect, optional
            Exact coordinates of the solved zone within the plane.
        pixels_to_mm_ratio : (sx, sy), optional
            Pixel-to-mm scale factors if you want to convert image pixels
            to real-world millimeters. Example: (0.25, 0.25) means
            each pixel equals 0.25 mm.
        """
        # Default values if not provided
        if base is None:
            base = Rect(x=0.0, y=0.0, w=450, h=450)
        if area_unsolved is None:
            area_unsolved = Rect(x=76.5, y=200, w=297.0, h=210)
        if area_solved is None:
            area_solved = Rect(x=120, y=20, w=210, h=148)

        # Assign to instance
        self.base = base
        self.area_unsolved = area_unsolved
        self.area_solved = area_solved
        self.ratiox, self.ratioy = pixels_to_mm_ratio
        self.unsolved_puzzles=[]
        self.solved_puzzles = []

    def _as_pts(self, contour):
        """Return (Nx2 float32 points, original_shape)."""
        arr = np.asarray(contour)
        orig_shape = arr.shape
        pts = arr.reshape(-1, 2).astype(np.float32)
        return pts, orig_shape

    def _from_pts(self, pts, orig_shape):
        """Reshape Nx2 points back to original contour shape, float32 for OpenCV."""
        return np.asarray(pts, dtype=np.float32).reshape(orig_shape)

        
    # Import a copy of the puzzle pieces
    def _import_puzzles(self, puzzles: List[Puzzle], target_list, img_height=964) -> None:
        target_list.clear()

        for p in puzzles:
            orig = np.asarray(p.contour)
            orig_shape = orig.shape

            # IMPORTANT: float32 for OpenCV contourArea
            pts = orig.reshape(-1, 2).astype(np.float32).copy()
            pts[:, 1] = np.float32(img_height) - pts[:, 1]   # invert Y

            cnt_global = pts.reshape(orig_shape)  # keeps (N,1,2)
            new_puzzle = Puzzle(cnt_global, p.index)
            #The [::-1] reverses the list of corners to match the order of the real puzzle pieces. 
            #This has to be done because in GlobalArea, the y axis gets flipped to follow the normal graph style.
            #It would be better to first assign the corners, then do the conversion. 
            new_puzzle.corners = new_puzzle.get_best_4_corners()[::-1]
            target_list.append(new_puzzle)






    def set_unsolved_puzzles(self, puzzles) -> None:
        self._import_puzzles(puzzles, self.unsolved_puzzles)


    def set_solved_puzzles(self, puzzles) -> None:
        self._import_puzzles(puzzles, self.solved_puzzles)




    def scale_all_puzzles(self, ratio_x: float, ratio_y: float) -> None:
        """
        Scale contours AND corners for both unsolved and solved puzzle lists.
        (Smaller number makes smaller contour)
        """
        for group in (self.unsolved_puzzles, self.solved_puzzles):
            for p in group:
                cnt = p.contour
                pts, orig_shape = self._as_pts(cnt)
                pts[:, 0] *= ratio_x
                pts[:, 1] *= ratio_y
                new_cnt = self._from_pts(pts, orig_shape)

                if hasattr(p, "set_contour"):
                    p.set_contour(new_cnt)
                else:
                    p.contour = new_cnt

                if hasattr(p, "corners") and p.corners is not None:
                    c = np.asarray(p.corners, dtype=float).reshape(4, 2)
                    c[:, 0] *= ratio_x
                    c[:, 1] *= ratio_y
                    p.corners = c
    
    def _translate_puzzles(self, target_list, dx: float, dy: float) -> None:
        """
        Translate all contours in target_list by (dx, dy).
        Mutates the list in-place (replaces each contour with a translated copy).
        """
        if not target_list:
            return

        translated = []
        for p in target_list:
            cnt = p.contour
            crn = p.corners
            pts = np.asarray(cnt).reshape(-1, 2).astype(float)
            pts[:, 0] += dx
            pts[:, 1] += dy
            p.contour = pts

            corner_pts = np.asarray(crn).reshape(-1, 2).astype(float)
            corner_pts[:, 0] += dx
            corner_pts[:, 1] += dy
            p.corners = corner_pts



    def translate_unsolved_puzzles(self, dx: float, dy: float) -> None:
        self._translate_puzzles(self.unsolved_puzzles, dx, dy)

    def translate_solved_puzzles(self, dx: float, dy: float) -> None:
        self._translate_puzzles(self.solved_puzzles, dx, dy)



    def get_unsolved_puzzle_piece(self,pos):
        return self.unsolved_puzzles[pos]

    def get_solved_puzzle_piece(self,pos):
        return self.solved_puzzles[pos]



    def get_matching_edge_lines(self, piece_a, edge_a, piece_b, edge_b, solved=True):
        """
        Return two edge lines (lineA, lineB), each as (p0, p1),
        using stored corners in order [TL, TR, BR, BL].

        Assumes edge indices: 0=top, 1=right, 2=bottom, 3=left.
        Assumes piece indices from Matching are 1-based.
        """
        puzzles = self.solved_puzzles
        EDGE_TO = ((0, 1), (1, 2), (2, 3), (3, 0))

        def edge_line(piece_id_1based, edge_idx):
            p = puzzles[piece_id_1based - 1]
            c = np.asarray(p.corners, dtype=np.float32).reshape(4, 2)
            i, j = EDGE_TO[edge_idx % 4]
            return c[i], c[j]

        return edge_line(piece_a, edge_a), edge_line(piece_b, edge_b)

    def check_corners_in_area(self,rect, pieces, eps=1e-6):
        '''
        
        :param rect: recxtangle area that should set the boundary. Solved or unsolved
        :param pieces: Puzzle pieces for that boundary area. Solved or unsolved
        :param eps: Amount that the coordinate can be off. 
        '''
        x_start, y_start, width, height = rect.x, rect.y, rect.w, rect.h
        left_bound = x_start
        right_bound = x_start+width
        bottom_bound = y_start
        top_bound = y_start + height
        for pi, p in enumerate(pieces):
            if not hasattr(p, "corners") or p.corners is None:
                raise ValueError(f"Piece {pi} has no corners set")
    
            crns = np.asarray(p.corners, dtype=float).reshape(-1, 2)
            for ci, (x, y) in enumerate(crns):
                if x < left_bound - eps or x > right_bound + eps or y < bottom_bound - eps or y > top_bound + eps:
                    # optional debug print:
                    # print(f"Out of bounds: piece={pi}, corner={ci}, (x,y)=({x:.2f},{y:.2f})")
                    return False
    
        return True
                


    # ------------------------------------------------------------------
    # Visualization
        # ------------------------------------------------------------------
    def show(self, invert_y: bool = False, margin: float = 10.0,
             title: str = "GlobalArea — Physical Coordinate Plane") -> None:
        from matplotlib.patches import Rectangle
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))

        def draw_rect(rect, color: str, label: str, lw: float = 2.0):
            ax.add_patch(Rectangle((rect.x, rect.y), rect.w, rect.h,
                                   fill=False, ec=color, lw=lw))
            ax.text(rect.x + 5, rect.y + 5, label, color=color,
                    fontsize=9, weight="bold")

        def draw_labeled_contour(p, idx, color, prefix):
            cnt= p.contour
            pts = np.asarray(cnt).reshape(-1, 2)
            closed = np.vstack([pts, pts[0]])
            ax.plot(closed[:, 0], closed[:, 1], '-', lw=1.5,
                    color=color, alpha=0.8)

            # Label at centroid
            center = pts.mean(axis=0)
            ax.text(center[0], center[1],
                    f"{prefix}{idx}",
                    color=color,
                    fontsize=10,
                    weight="bold",
                    ha="center",
                    va="center")
        def draw_corners(self,puzzle, piece_idx, color, prefix):
            corners = puzzle.corners
            pts = np.asarray(corners, dtype=float).reshape(-1, 2)
            ax.scatter(pts[:, 0], pts[:, 1], s=25, marker='o', color=color, zorder=5)

            # Label each corner 0..3 (your TL/TR/BR/BL order from get_best_4_corners)
            for k, (x, y) in enumerate(pts):
                ax.text(x + 1.5, y + 1.5, f"{prefix}{piece_idx}.{k}",
                        color=color, fontsize=8, weight="bold",
                        ha="left", va="bottom")

        # Draw geometry
        draw_rect(self.base, 'black', 'BASE')
        draw_rect(self.area_unsolved, 'tab:blue', 'UNSOLVED')
        draw_rect(self.area_solved, 'tab:green', 'SOLVED')

        # Draw unsolved contours with indices
        if self.unsolved_puzzles:
            for i, p in enumerate(self.unsolved_puzzles):
                draw_labeled_contour(p, i, color='tab:red', prefix="U")

        # Draw solved contours with indices
        if self.unsolved_puzzles:
            for i, p in enumerate(self.solved_puzzles):
                draw_labeled_contour(p, i, color='tab:orange', prefix="S")
        
        # Draw unsolved corners
#        if self.unsolved_puzzles:
#            for i, p in enumerate(self.unsolved_puzzles):
#                draw_corners(self,p, i, color="tab:purple", prefix="UC")
#        
#        # Draw solved corners
#        if self.unsolved_puzzles:
#            for i, p in enumerate(self.solved_puzzles):
#                draw_corners(self,p, i, color="tab:purple", prefix="SC")

        ax.set_xlim(self.base.x - margin, self.base.right + margin)
        ax.set_ylim(self.base.y - margin, self.base.bottom + margin)

        # Aspect & orientation
        ax.set_aspect('equal', adjustable='box')
        if invert_y:
            ax.invert_yaxis()

        # Cosmetics
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)' if not invert_y else 'Y (mm, down)')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
