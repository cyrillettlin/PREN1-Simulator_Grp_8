from __future__ import print_function
from puzzle import Puzzle

import os
import cv2 as cv
import numpy as np


class EdgeDetection:
    MIN_AREA = 900
    
    def __init__(self, path_to_file: str):
        self.path_to_file = os.path.normpath(path_to_file)
        self.src = None
        self.src_gray = None
        self.edges = None
        self.contours = None
        self.largest_contours = None
        self.puzzle_pieces = []

    def load(self):
        """ load image from path in main"""
        self.src = cv.imread(self.path_to_file)
        if self.src is None:
            raise FileNotFoundError(f"Datei nicht gefunden: {self.path_to_file}")
        self.src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        return self
        
    def find_contours(self):
        """Find contours using Gaussian blur + Otsu threshold."""
        img_blur = cv.GaussianBlur(self.src_gray, (5,5), 0)
        _, img_thresh = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        cv.imshow('thresh', img_thresh)

        contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        return contours


    def filter_contours(self, min_area = MIN_AREA):
        """Filters contours that are too small based on the min_area"""
        if self.contours is None or len(self.contours) == 0:
            raise ValueError("Keine Konturen gefunden. Bitte zuerst find_contours() aufrufen.")
        
        filtered_contours = [c for c in self.contours if cv.contourArea(c) >= min_area]
        self.puzzle_pieces = [Puzzle(cnt, i + 1) for i, cnt in enumerate(filtered_contours)]
        self.contours = filtered_contours
        return filtered_contours

        
    def get_puzzle_pieces(self):
        return self.puzzle_pieces

    def show_result(self):
        """
        Zeigt die vom Puzzle-Objekt erkannten Edges (aus get_puzzle_edges)
        farblich markiert mit den entsprechenden Ecken.
        """
        output = self.src.copy()

        # Farben für die vier Edges (top, right, bottom, left)
        edge_colors = [
            (0, 255, 255),   # Gelb
            (255, 0, 255),   # Magenta
            (255, 255, 0),   # Cyan
            (0, 165, 255)    # Orange
        ]

        for piece in self.puzzle_pieces:
            # --- Erhalte Edges aus Puzzle.py ---
            edges = piece.get_puzzle_edges()

            # --- Zeichne jede Edge mit eigener Farbe ---
            for i, edge_points in enumerate(edges):
                if edge_points and len(edge_points) > 1:
                    pts_array = np.array(edge_points, dtype=np.int32)
                    cv.polylines(output, [pts_array], isClosed=False, color=edge_colors[i % 4], thickness=2)

            # --- Ecken markieren ---
            corners = piece.get_best_4_corners()
            for j, corner in enumerate(corners):
                cv.circle(output, corner, 6, (0, 255, 255), -1)
                cv.putText(output, f"{j+1}",
                        (corner[0] + 5, corner[1] - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Bild verkleinern für Übersicht
        scaled_output = cv.resize(output, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)

        cv.imshow("Detected Puzzle Edges", scaled_output)
        cv.waitKey(0)
        cv.destroyAllWindows()


# Aktuelle Probleme bei Erkennung:
# Gewisse Puzzle Objekte erkennen eine zu kleine Area andere eine zu grosse (
# z.B Real-Puzzle_2 erkennt Schraubenwinde aber Puzzle 4 nicht) --> Möglicher SimalarityCheck mit
# Puzzlevorlage

# Ecken werden nicht zuverlässig erkennt --> Recherche bessere Lösungsansätze

