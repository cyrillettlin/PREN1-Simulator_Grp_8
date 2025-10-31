from __future__ import print_function
from puzzle import Puzzle

import os
import cv2 as cv
import numpy as np


class EdgeDetection:
    
    def __init__(self, path_to_file: str):
        self.path_to_file = os.path.normpath(path_to_file)
        #Definiere Threshold
        self.low_threshold = 100
        self.high_threshold = 300
        self.src = None
        self.src_gray = None
        self.edges = None
        self.contours = None
        self.largest_contours = None
        self.puzzle_pieces = []

    def load(self):
        #Bild laden
        self.src = cv.imread(self.path_to_file)
        if self.src is None:
            raise FileNotFoundError(f"Datei nicht gefunden: {self.path_to_file}")
        self.src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        
    def find_contours(self):
        """Kanten- und Konturenerkennung"""

        # Kanten finden
        img_blur = cv.blur(self.src_gray, (3,3))
        self.edges = cv.Canny(img_blur, self.low_threshold, self.high_threshold, 3)

        # Konturen finden 
        contours, _ = cv.findContours(self.edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        return self.contours

    
    def separate_contours(self):
        if not self.contours:
            raise ValueError("Keine Konturen gefunden. Bitte zuerst find_contours() aufrufen.")
        self.contours = sorted(self.contours, key=cv.contourArea, reverse=True)
        #Vier grösste Konturen
        self.largest_contours = self.contours[:4]

    def create_puzzle(self):
        self.puzzle_pieces = [puzzle(cnt, i + 1) for i, cnt in enumerate(self.largest_contours)]
        for piece in puzzle_pieces:
            print(piece)



#Output prov.
    def show_result(self):
            """Zeigt das Originalbild mit Konturen und Infos an."""
            output = self.src.copy()
            colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255)]

            for i, cnt in enumerate(self.largest_contours):
                color = colors[i % len(colors)]
                cv.drawContours(output, [cnt], -1, color, 3)
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    area = cv.contourArea(cnt)
                    cv.putText(output, f"#{i+1} ({int(area)}px)", (cx - 40, cy + 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv.putText(output, f"Low: {self.low_threshold}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(output, f"High: {self.high_threshold}", (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.imshow("Vier groesste Konturen", output)
            cv.waitKey(0)
            cv.destroyAllWindows()
