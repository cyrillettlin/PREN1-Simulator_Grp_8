import os
import cv2 as cv
import numpy as np
import logging
from puzzle import Puzzle

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class EdgeDetection:
    MIN_AREA = 50000 #in Quadratpixeln, je nach Bildauflösung vergrössern (siehe Logs), für filter_contours
    
    def __init__(self, path_to_file: str):
        self.path_to_file = os.path.normpath(path_to_file)
        self.src = None
        self.src_gray = None
        self.contours = None
        self.puzzle_pieces = []

    def load(self):
        """Load image from path"""
        self.src = cv.imread(self.path_to_file)
        if self.src is None:
            raise FileNotFoundError(f"Datei nicht gefunden: {self.path_to_file}")
        self.src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        logging.info(f"Bild geladen: {self.path_to_file} Größe={self.src.shape[1]}x{self.src.shape[0]}")
        return self
        
    def find_contours(self):
        """Find contours using Gaussian blur + Otsu threshold."""
        img_blur = cv.GaussianBlur(self.src_gray, (9,9), 0) #ksize 5,5, muss jeh nach Schatten / Rauschen im Bild angepasst werden
        _, img_thresh = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        logging.info(f"Gefundene Konturen: {len(contours)}")
        return contours

    def filter_contours(self, min_area = MIN_AREA):
        """Filter contours by area and create Puzzle objects"""
        if self.contours is None or len(self.contours) == 0:
            raise ValueError("Keine Konturen gefunden. Bitte zuerst find_contours() aufrufen.")
        
        filtered_contours = [c for c in self.contours if cv.contourArea(c) >= min_area]
        self.puzzle_pieces = [Puzzle(cnt, i + 1) for i, cnt in enumerate(filtered_contours)]
        self.contours = filtered_contours
        logging.info(f"Anzahl Puzzle-Teile nach Filterung: {len(filtered_contours)}")
        return filtered_contours

    def get_puzzle_pieces_infos(self):
        """logging of puzzle pieces information to adjust MIN_AREA"""
        for p in self.puzzle_pieces:
            x, y, w, h = p.bounding_box
            logging.info(f"  Teil {p.index}: Fläche={p.area:.2f}, Box=({x},{y},{w},{h})")

    def get_puzzle_pieces(self):
        """creates instances of puzzle pieces"""
        return self.puzzle_pieces
