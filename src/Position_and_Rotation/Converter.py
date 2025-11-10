

class Converter:
    def __init__(self):
        pass

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
