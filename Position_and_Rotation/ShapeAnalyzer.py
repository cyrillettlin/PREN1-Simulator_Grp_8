from __future__ import annotations
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class ShapeAnalyzer:
    """
    Analyze shapes in an image using OpenCV.

    - load()                 → loads image and grayscale
    - detect_edges()         → Canny edges
    - find_contours()        → find contours from edges
    - top_n_contours(n)      → keep n largest contours
    - annotate()             → draw contours + boxes + labels
    - metrics()              → area, perimeter, bbox for each contour
    - similarity_matrix()    → pairwise shape similarity
    - show()/save()          → visualize/save annotated result
    """

    def __init__(
        self,
        image_path: str | Path,
        canny_low: int = 50,
        canny_high: int = 150,
        retrieval_mode: int = cv.RETR_EXTERNAL,
        approx_mode: int = cv.CHAIN_APPROX_SIMPLE,
    ):
        self.image_path = Path(image_path)
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.retrieval_mode = retrieval_mode
        self.approx_mode = approx_mode

        self.src: Optional[np.ndarray] = None
        self.src_gray: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        self.contours: List[np.ndarray] = []
        self.hierarchy: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    # ---- Pipeline steps -----------------------------------------------------

    def load(self) -> "ShapeAnalyzer":
        """Load image and convert to grayscale."""
        self.src = cv.imread(str(self.image_path))
        if self.src is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")
        self.src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        self.output = self.src.copy()
        return self

    def detect_edges(self) -> "ShapeAnalyzer":
        """Detect edges using Canny."""
        if self.src_gray is None:
            raise RuntimeError("Call load() before detect_edges().")
        self.edges = cv.Canny(self.src_gray, self.canny_low, self.canny_high)
        return self

    def find_contours(self) -> "ShapeAnalyzer":
        """Find contours from edges."""
        if self.edges is None:
            raise RuntimeError("Call detect_edges() before find_contours().")
        contours, hierarchy = cv.findContours(self.edges, self.retrieval_mode, self.approx_mode)
        self.contours = contours
        self.hierarchy = hierarchy
        return self

    def top_n_contours(self, n: int = 10) -> "ShapeAnalyzer":
        """Keep only the n largest contours by area."""
        self.contours = sorted(self.contours, key=cv.contourArea, reverse=True)[:n]
        return self


    # ---- Analysis helpers ---------------------------------------------------

    def metrics(self, contours: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """Return metrics for each contour: area, perimeter, bbox."""
        cs = contours if contours is not None else self.contours
        out = []
        for i, c in enumerate(cs):
            area = float(cv.contourArea(c))
            perimeter = float(cv.arcLength(c, True))
            x, y, w, h = cv.boundingRect(c)
            out.append({
                "index": i,
                "area": area,
                "perimeter": perimeter,
                "bbox": (x, y, w, h),
            })
        return out

    def similarity_matrix(
        self,
        method: int = cv.CONTOURS_MATCH_I1
    ) -> np.ndarray:
        """Compute pairwise shape similarity for current contours."""
        n = len(self.contours)
        sim = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                sim[i, j] = cv.matchShapes(self.contours[i], self.contours[j], method, 0.0)
        return sim

    # ---- Visualization ------------------------------------------------------

    def annotate(
        self,
        draw_contours: bool = True,
        draw_boxes: bool = True,
        draw_labels: bool = True,
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        box_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> "ShapeAnalyzer":
        """Draw contours, bounding boxes, and labels on the output image."""
        if self.output is None:
            raise RuntimeError("Call load() first.")
        if draw_contours:
            cv.drawContours(self.output, self.contours, -1, contour_color, thickness)

        if draw_boxes or draw_labels:
            for i, c in enumerate(self.contours):
                x, y, w, h = cv.boundingRect(c)
                if draw_boxes:
                    cv.rectangle(self.output, (x, y), (x + w, y + h), box_color, thickness)
                if draw_labels:
                    cv.putText(
                        self.output,
                        f"Objekt {i+1}",
                        (x, max(0, y - 10)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        1,
                        cv.LINE_AA
                    )
        return self

    def show(self, window_name: str = "Objekte", wait: int = 0) -> None:
        """Show annotated image."""
        if self.output is None:
            raise RuntimeError("Nothing to show. Did you call annotate()? or load()?")
        cv.imshow(window_name, self.output)
        cv.waitKey(wait)
        cv.destroyAllWindows()

    def save(self, path: str | Path) -> Path:
        """Save annotated image to disk."""
        if self.output is None:
            raise RuntimeError("Nothing to save. Did you call annotate()? or load()?")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(path), self.output)
        return path

    # ---- Convenience one-shot ----------------------------------------------

    def process(self, top_n: int = 4) -> "ShapeAnalyzer":
        """Run the full pipeline with default steps."""
        return (
            self.load()
                .detect_edges()
                .find_contours()
                .top_n_contours(top_n)
                .annotate()
        )

