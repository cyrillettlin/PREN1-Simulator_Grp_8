from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


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
        self.contours= []
        self.solved_contours =[]
        

    def set_contour(self, puzzles) -> None:
        """
        Import contours from puzzles and invert the Y-axis
        so that (0,0) is bottom-left instead of top-left.
        """
        img_height = 964
        self.contours = []  # reset list
        for p in puzzles:
            cnt = np.asarray(p.contour).reshape(-1, 2).astype(float)
            # Invert Y axis
            cnt[:, 1] = img_height - cnt[:, 1]
            self.contours.append(cnt)
    
    def set_solved_contours(self, puzzles) -> None:
        """
        Import contours from puzzles and invert the Y-axis
        so that (0,0) is bottom-left instead of top-left.
        """
        img_height = 964
        self.contours = []  # reset list
        for p in puzzles:
            cnt = np.asarray(p.contour).reshape(-1, 2).astype(float)
            # Invert Y axis
            cnt[:, 1] = img_height - cnt[:, 1]
            self.contours.append(cnt)

        

    def scale_contours(self,ratio_x,ratio_y)-> None:
        """
        Scales the contours by the given Ratio.
        Smaller Number makes it smaller. Larger makes larger.
        """
        scaled =[]
        
        ratio_x = ratio_x
        ratio_y = ratio_y
        contours = self.contours
        for cnt in contours:
            pts = cnt.reshape(-1, 2).astype(float)
            pts = pts * [ratio_x, ratio_y]  # scale
            scaled.append(pts)
        self.contours = scaled
    
    def translate_contours(self, dx: float, dy: float) -> None:
        """
        Translate all contours in the global coordinate system
        by a given (dx, dy) offset in millimeters.

        Parameters
        ----------
        dx : float
            Translation distance along X-axis (mm). Positive → right.
        dy : float
            Translation distance along Y-axis (mm). Positive → up.
        """
        if not hasattr(self, "contours") or not self.contours:
            print("No contours to translate.")
            return

        translated = []
        for cnt in self.contours:
            pts = np.asarray(cnt).reshape(-1, 2).astype(float)
            pts[:, 0] += dx
            pts[:, 1] += dy
            translated.append(pts)

        self.contours = translated



    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def show(self, invert_y: bool = False, margin: float = 10.0, title: str = "GlobalArea — Physical Coordinate Plane") -> None:
        """
        Display all defined rectangles on the physical coordinate plane.

        Args:
            invert_y: If True, flips Y so it increases downward (image-style).
            margin:   Extra space (in mm) added around the base rectangle in the view.
            title:    Figure title.
        """

        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(7, 5))

        def draw_rect(rect: Rect, color: str, label: str, lw: float = 2.0):
            ax.add_patch(Rectangle((rect.x, rect.y), rect.w, rect.h, fill=False, ec=color, lw=lw))
            ax.text(rect.x + 5, rect.y + 5, label, color=color, fontsize=9, weight="bold")

        # Draw geometry
        draw_rect(self.base, 'black', 'BASE')
        draw_rect(self.area_unsolved, 'tab:blue', 'UNSOLVED')
        draw_rect(self.area_solved, 'tab:green', 'SOLVED')

        # --- Draw contours if available ---
        if hasattr(self, "contours") and self.contours:
            for cnt in self.contours:
                pts = np.asarray(cnt).reshape(-1, 2)
                closed = np.vstack([pts, pts[0]])  # close contour for plotting
                ax.plot(closed[:, 0], closed[:, 1],
                        '-', lw=1.5, color='tab:red', alpha=0.8)


        # Set axis limits to the base rectangle extents (so we don't get 0..1)
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


if __name__ == "__main__":

    
    # Optional: you know that each camera pixel = 0.25 mm in both axes
    pixels_to_mm_ratio = (0.25, 0.25)

    ga = GlobalArea(pixels_to_mm_ratio=pixels_to_mm_ratio)

    # Show the layout on the coordinate plane
    ga.show(invert_y=False)
