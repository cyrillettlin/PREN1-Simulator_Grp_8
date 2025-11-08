from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt


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
        base: Rect,
        area_unsolved: Rect,
        area_solved: Rect,
        *,
        pixels_to_mm_ratio: Optional[Tuple[float, float]] = (0.25,0.25)
    ) -> None:
        """
        Parameters
        ----------
        base : Rect
            The total working area (absolute position and size).
        area_unsolved : Rect
            Exact coordinates of the unsolved zone within the plane.
        area_solved : Rect
            Exact coordinates of the solved zone within the plane.
        pixels_to_mm_ratio : (sx, sy), optional
            Pixel-to-mm scale factors if you want to convert image pixels
            to real-world millimeters. Example: (0.25, 0.25) means
            each pixel equals 0.25 mm.
        """
        self.base = base
        self.area_unsolved = area_unsolved
        self.area_solved = area_solved
        self.ratiox, self.ratioy = pixels_to_mm_ratio

    # --------------------------------------------------
    # Conversion methods (using per-axis ratios)
    # --------------------------------------------------
    def px_to_mm(self, u_px: float, v_px: float) -> tuple[float, float]:
        """
        Convert image pixel coordinates (u_px, v_px) to millimeters (x_mm, y_mm)
        using the X and Y pixel-to-mm ratios defined for this GlobalArea.

        Parameters
        ----------
        u_px, v_px : float
            Pixel coordinates (e.g., from a contour centroid).

        Returns
        -------
        (x_mm, y_mm) : tuple of float
            Converted coordinates in millimeters.
        """
        x_mm = u_px * self.sx
        y_mm = v_px * self.sy
        return x_mm, y_mm

    def mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """
        Convert millimeter coordinates (x_mm, y_mm) back to image pixels (u_px, v_px)
        using the X and Y pixel-to-mm ratios defined for this GlobalArea.

        Parameters
        ----------
        x_mm, y_mm : float
            Physical coordinates in millimeters.

        Returns
        -------
        (u_px, v_px) : tuple of float
            Converted coordinates in pixels.
        """
        u_px = x_mm / self.sx
        v_px = y_mm / self.sy
        return u_px, v_px


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
    # Base board is 600 × 400 mm
    base = Rect(x=0.0, y=0.0, w=600.0, h=400.0)

    # Unsolved zone occupies the left side of the table
    area_unsolved = Rect(x=20.0, y=20.0, w=260.0, h=360.0)

    # Solved zone is on the right side, with a 40 mm gap between them
    area_solved = Rect(x=320.0, y=120.0, w=260.0, h=260.0)

    # Optional: you know that each camera pixel = 0.25 mm in both axes
    pixels_to_mm_ratio = (0.25, 0.25)

    ga = GlobalArea(
        base=base,
        area_unsolved=area_unsolved,
        area_solved=area_solved,
        pixels_to_mm_ratio=pixels_to_mm_ratio,
    )

    # Show the layout on the coordinate plane
    ga.show(invert_y=False)
