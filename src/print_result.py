from __future__ import print_function
import cv2 as cv
import numpy as np


def print_result(self):
    """
    Zeigt die vom Puzzle-Objekt erkannten Edges (aus get_puzzle_edges)
    farblich markiert mit den entsprechenden Ecken.
    """
    output = self.src.copy()

    # Farben für die vier Edges (top, right, bottom, left)
    edge_colors = [
        (0, 255, 255),  # Gelb
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (0, 165, 255)  # Orange
    ]

    for piece in self.puzzle_pieces:
        # --- Erhalte Edges aus Puzzle.py ---
        edges = piece.get_puzzle_edges()

        # --- Zeichne jede Edge mit eigener Farbe ---
    for i, edge in enumerate(edges):
        pts = edge.get("points", [])
        if pts and len(pts) > 1:
            pts_array = np.array(pts, dtype=np.int32)
            cv.polylines(
                output,
                [pts_array],
                isClosed=False,
                color=edge_colors[i % 4],
                thickness=2
            )


        # --- Ecken markieren ---
        corners = piece.get_best_4_corners()
        for j, corner in enumerate(corners):
            cv.circle(output, corner, 6, (0, 255, 255), -1)
            cv.putText(output, f"{j + 1}",
                       (corner[0] + 5, corner[1] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    scaled_output = cv.resize(output, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
    cv.imshow("Detected Puzzle Edges", scaled_output)
    cv.waitKey(0)
    cv.destroyAllWindows()
