import logging

import cv2 as cv
import numpy as np
from typing import List, Tuple, Dict, Optional

class Puzzle:

    def __init__(self, contour, index):
        self.index = index
        self.contour = contour
        self.area = cv.contourArea(contour)
        self.center_point = self.get_center_point()

        #bounding box
        # nur für das Logging
        self.rect_simple = cv.boundingRect(contour) #boundingbox parallel zum Bildrahmen
        # für mathematische Berechnungen
        self.rect_rotated = cv.minAreaRect(contour) #rotierte boundingbox
        # für Abwärtskompatibilität für Logging Skript im Code
        self.bounding_box = self.rect_simple

        #Caching, für Performance
        self.corners: Optional[List[Tuple[int, int]]] = None
        self.edges: Optional[List[Dict]] = None

        # Initiales Logging für das Puzzleteil
        x, y, w, h = self.rect_simple
        logging.info(
            f"PuzzlePiece {self.index} initialisiert: "
            f"Fläche={self.area:.1f}px², "
            f"Schwerpunkt={self.center_point}, "
            f"Box=(x:{x}, y:{y}, w:{w}, h:{h})"
        )


    def get_contour(self):
        return self.contour

    def set_contour(self,cnt):
        self.contour =cnt

    @staticmethod
    def get_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Berechnet den Innenwinkel bei p2 in Grad"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        angle = np.abs(np.degrees(a1 - a2))
        return angle if angle <= 180 else 360 - angle

    @staticmethod
    def is_convex(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        """prüft via Kreuzprodukt ob Ecke p2 nach aussen geht, res <0 muss eventuell angepasst werden, jeh nach dem ob Speicherung im Uhrzeigersinn oder Gegenuhrzeigersinn"""
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        return (v1[0] * v2[1] - v1[1] * v2[0]) < 0

    def get_best_4_corners(self, epsilon_factor: float =0.04) -> List[Tuple[int, int]]: #epsilon_factor=0.00002, muss je nach Bildqualität angepasst werden
            """Extrahiert die 4 markantesten Ecken basierend auf der rotierten Bounding Box"""
            #Rauschen reduzieren
            #Cache
            if self.corners is not None:
                return self.corners

            # 1. Kontur vereinfachen
            epsilon = epsilon_factor * cv.arcLength(self.contour, True)
            approx = cv.approxPolyDP(self.contour, epsilon, True).reshape(-1, 2)

            # 2. Kandidaten filtern (Winkel & Konvexität)
            candidates = []
            n = len(approx)
            for i in range(n):
                angle = self.get_angle(approx[i-1], approx[i], approx[(i+1)%n])
                if 70 <= angle <= 115 and self.is_convex(approx[i-1], approx[i], approx[(i+1)%n]):
                 candidates.append(approx[i])

            # Falls zu wenig markante Ecken, nimm die am weitesten vom Zentrum entfernten Punkte
            search_pool = np.array(candidates) if len(candidates) >= 4 else approx

            # 3. Abgleich mit rotierter Bounding Box
            box_points = np.int32(cv.boxPoints(self.rect_rotated))
            real_corners = []

            for bp in box_points:
                # Vektorisierte Distanzberechnung
                dists = np.linalg.norm(search_pool - bp, axis=1)
                real_corners.append(tuple(search_pool[np.argmin(dists)]))

            # 4. Sortierung: Oben-Links, Oben-Rechts, Unten-Rechts, Unten-Links
            real_corners = sorted(real_corners, key=lambda p: p[1])
            top = sorted(real_corners[:2], key=lambda p: p[0])
            bottom = sorted(real_corners[2:], key=lambda p: p[0], reverse=True)

            self.corners = top + bottom

            logging.info(
                f"  Teil {self.index} Ecken gefunden: "
                f"Oben-Links={self.corners[0]}, Oben-Rechts={self.corners[1]}, "
                f"Unten-Rechts={self.corners[2]}, Unten-Links={self.corners[3]}"
            )

            return self.corners


    def get_puzzle_edges(self) -> List[Dict]:
        """
        Robuste Extraktion der Kontursegmente zwischen den 4 Ecken.
        Rückgabe: [top_edge, right_edge, bottom_edge, left_edge]
        Jede Edge ist eine Liste von (x,y)-Tupeln entlang der Kontur.
        """
        if self.edges is not None:
            return self.edges

        contour_pts = self.contour.reshape(-1, 2)
        n_pts = len(contour_pts)
        corners = self.get_best_4_corners()

        if n_pts == 0 or len(corners) != 4:
            logging.warning(f"Teil {self.index}: Ecken konnten nicht bestimmt werden.")
            return [{"points": [], "type": "inner"} for _ in range(4)]

        corner_indices = []
        for c in corners:
            dists = np.linalg.norm(contour_pts - np.array(c), axis=1)
            corner_indices.append(np.argmin(dists))

        corner_indices.sort()

        # Segmente extrahieren
        raw_segments = []
        for i in range(4):
            idx1 = corner_indices[i]
            idx2 = corner_indices[(i + 1) % 4]

            if idx2 < idx1:
                seg = np.vstack((contour_pts[idx1:], contour_pts[:idx2 + 1]))
            else:
                seg = contour_pts[idx1:idx2 + 1]

            raw_segments.append([tuple(p) for p in seg])

        # Segmente geometrisch analysieren und der richtigen Himmelsrichtung zuweisen
        cx, cy = self.center_point
        classified = {"top": [], "right": [], "bottom": [], "left": []}

        for seg in raw_segments:
            if not seg:
                continue
            seg_arr = np.array(seg)
            # Mittelpunkt des Segments berechnen
            mx, my = np.mean(seg_arr, axis=0)
            # Vektor vom Schwerpunkt zum Segment-Mittelpunkt
            dx, dy = mx - cx, my - cy

            # Geometrische Zuordnung basierend auf der dominanten Achse
            if abs(dx) > abs(dy):
                label = "right" if dx > 0 else "left"
            else:
                label = "bottom" if dy > 0 else "top"

            classified[label] = seg

        self.edges = [
            {"points": classified["top"], "side": "top"},
            {"points": classified["right"], "side": "right"},
            {"points": classified["bottom"], "side": "bottom"},
            {"points": classified["left"], "side": "left"}
        ]

        logging.info(
            f"  Teil {self.index} Kanten geometrisch geordnet (Pixel): "
            f"Top={len(classified['top'])}, Right={len(classified['right'])}, "
            f"Bottom={len(classified['bottom'])}, Left={len(classified['left'])}"
        )

        return self.edges

    def get_center_point(self) -> Tuple[int, int]:
        """calculate the center point of the puzzle with moments"""
        m = cv.moments(self.contour)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        return (0, 0)

    def __repr__(self) -> str:
        return f"PuzzlePiece(id={self.index}, area={self.area:.1f})"


    

   
