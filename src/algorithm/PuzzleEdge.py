import numpy as np
from typing import List

class PuzzleEdge:
    """
    einzelne Kante eines Puzzleteils
    Konturpunkte (2D), Länge und normierte 1D-Form
    """

    def __init__(self, points: np.ndarray):
        self.points = np.array(points, dtype=np.float32)
        self.length = self._compute_length()
        self.normalized_shape = self._compute_normalized_shape()

    def _compute_length(self) -> float:
        diffs = np.diff(self.points, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        return float(np.sum(seg_lengths))

    def _compute_normalized_shape(self, num_points: int = 200) -> np.ndarray:
        contour = self.points
        if len(contour) < 2:
            return np.zeros(num_points)

        cumulative = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
        cumulative = np.insert(cumulative, 0, 0)
        target = np.linspace(0, cumulative[-1], num_points)
        interp_x = np.interp(target, cumulative, contour[:, 0])
        interp_y = np.interp(target, cumulative, contour[:, 1])

        dx, dy = contour[-1] - contour[0]
        theta = np.arctan2(dy, dx)
        rot_matrix = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])
        rotated = np.dot(np.stack([interp_x, interp_y], axis=1), rot_matrix.T)

        y_shape = rotated[:, 1]
        y_shape -= np.mean(y_shape)
        return y_shape

    def __repr__(self):
        return f"PuzzleEdge(length={self.length:.2f}, points={len(self.points)})"
    