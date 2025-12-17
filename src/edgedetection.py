"""
Detection of the puzzle piece
"""

from __future__ import print_function
from puzzle import Puzzle

import os
import numpy as np
import cv2 as cv

class EdgeDetection:
    MIN_AREA = 500
    
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

        # Bild für Zwischenschritte ausgeben
        scaled_output = cv.resize(img_thresh, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        cv.imshow('thresh, contours', scaled_output)

        contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        return contours


    def filter_contours(self, min_area = MIN_AREA):
        """Filters contours that are too small based on the min_area"""
        if self.contours is None or len(self.contours) == 0:
            raise ValueError("Keine Konturen gefunden. Bitte zuerst find_contours() aufrufen.")
        
        filtered_contours = [c for c in self.contours if cv.contourArea(c) >= min_area]

        # Bild für Zwischenschritte ausgeben
        output = self.src.copy()
        cv.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)
        scaled_output = cv.resize(output, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        cv.imshow('filtered_contours', scaled_output)

        self.puzzle_pieces = [Puzzle(cnt, i + 1) for i, cnt in enumerate(filtered_contours)]
        self.contours = filtered_contours
        return filtered_contours

        
    def get_puzzle_pieces(self):
        return self.puzzle_pieces
    
    def show_all_edges(self):
        """
        Zeigt alle erkannten Kanten aller Puzzle-Teile farblich an.
        """
        colors = [
            (0, 255, 255),  # top - gelb
            (255, 0, 255),  # right - magenta
            (255, 255, 0),  # bottom - cyan
            (0, 165, 255)   # left - orange
        ]

        for piece in self.puzzle_pieces:
            x, y, w, h = piece.bounding_box
            canvas = np.zeros((h + 20, w + 20, 3), dtype=np.uint8)

            edges = piece.get_puzzle_edges()
            for i, edge in enumerate(edges):
                pts = edge.get("points", [])
                if pts:
                    pts_rel = [(p[0]-x+10, p[1]-y+10) for p in pts]
                    cv.polylines(canvas, [np.array(pts_rel, dtype=np.int32)], isClosed=False, color=colors[i % 4], thickness=2)

            # Ecken markieren
            corners = piece.get_best_4_corners()
            for j, c in enumerate(corners):
                cv.circle(canvas, (c[0]-x+10, c[1]-y+10), 5, (0, 255, 255), -1)
                cv.putText(canvas, f"{j+1}", (c[0]-x+12, c[1]-y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            cv.imshow(f"Puzzle {piece.index} Edges", canvas)

        cv.waitKey(0)
        cv.destroyAllWindows()

