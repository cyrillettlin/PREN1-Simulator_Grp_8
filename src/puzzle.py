"""
This file contains the geometry and analysis function.
"""


import cv2 as cv
import numpy as np

class Puzzle:

    def __init__(self, contour, index):
        self.index = index
        self.contour = contour
        self.area = cv.contourArea(contour)
        self.bounding_box = cv.boundingRect(contour)
        self.center_point = self.get_center_point()

    def get_best_4_corners(self, epsilon_factor=0.00002):
            # Kontur approximieren, um Rauschen zu reduzieren und echte Eckpunkte zu finden
            # contour_arr = self.contour.reshape(-1, 2)
            epsilon = epsilon_factor * cv.arcLength(self.contour, True)
            approx = cv.approxPolyDP(self.contour, epsilon, True)
            approx_arr = approx.reshape(-1, 2)

            rect = cv.minAreaRect(self.contour) #umschliesst die Form als Rechteck, Rechteck mit der kleinsten Fläche
            box = cv.boxPoints(rect)
            box = np.int32(box)

            # Kontur für Berechnungen flachklopfen (N, 2) statt (N, 1, 2)
            # contour_arr = self.contour.reshape(-1, 2)

            real_corners = []

            # Für jede der 4 Box-Ecken den nächstgelegenen Punkt auf der Kontur finden
            for box_point in box:
                deltas = approx_arr - box_point
                dists = np.linalg.norm(deltas, axis=1)

                # Finde den Index des Minimums
                min_idx = np.argmin(dists)

                # Füge den echten Konturpunkt hinzu
                closest_point = tuple(approx_arr[min_idx])
                real_corners.append(closest_point)


            real_corners = sorted(real_corners, key=lambda p: p[1]) # Erst nach Y sortieren

            # Top-Gruppe (kleines Y) und Bottom-Gruppe (großes Y) unterscheiden
            top_group = sorted(real_corners[:2], key=lambda p: p[0]) # Nach X sortieren
            bottom_group = sorted(real_corners[2:], key=lambda p: p[0], reverse=True) # Nach X sortieren (Reverse für Uhrzeigersinn)

            sorted_corners = top_group + bottom_group

            return sorted_corners

    def get_puzzle_edges(self):
        """
        Robuste Extraktion der Kontursegmente zwischen den 4 Ecken.
        Rückgabe: [top_edge, right_edge, bottom_edge, left_edge]
        Jede Edge ist eine Liste von (x,y)-Tupeln entlang der Kontur.
        """
        contour_pts = self.contour.reshape(-1, 2)
        n = len(contour_pts)
        corners = self.get_best_4_corners()

        # Sicherheitschecks
        if n == 0 or len(corners) != 4:
            return [[], [], [], []]

        # Für jede Ecke Indices der Kontur nach Distanz sortiert
        corner_candidate_indices = []
        for c in corners:
            dists = np.linalg.norm(contour_pts - np.array(c), axis=1)
            sorted_idx = np.argsort(dists)
            corner_candidate_indices.append(list(sorted_idx))

        # Wähle für jede Ecke den nächstgelegenen, noch nicht verwendeten Index.
        used = set()
        assigned = [None] * 4
        for i in range(4):
            for idx in corner_candidate_indices[i]:
                if idx not in used:
                    assigned[i] = int(idx)
                    used.add(idx)
                    break
            # Falls alle nahe Indices bereits verwendet (sehr selten), suche mit Offset
            if assigned[i] is None:
                # suche vorwärts/backwärts vom absolut nächstliegenden Index
                base = corner_candidate_indices[i][0]
                found = False
                for offset in range(1, n):
                    cand = (base + offset) % n
                    if cand not in used:
                        assigned[i] = int(cand)
                        used.add(cand)
                        found = True
                        break
                    cand = (base - offset) % n
                    if cand not in used:
                        assigned[i] = int(cand)
                        used.add(cand)
                        found = True
                        break
                if not found:
                    # als letzten Ausweg benutze base (auch wenn bereits verwendet)
                    assigned[i] = int(base)

        # Wir haben jetzt für jede Ecke einen Index, aber die Reihenfolge der Ecken
        idx_corner_pairs = list(zip(assigned, corners))  # (index, (x,y))
        idx_corner_pairs.sort(key=lambda x: x[0])        # Kontur-Reihenfolge

        # 4) Extrahiere Segmente zwischen aufeinanderfolgenden Indices (wrap around)
        segments = []
        for i in range(4):
            idx1 = idx_corner_pairs[i][0]
            idx2 = idx_corner_pairs[(i + 1) % 4][0]
            # wrap-around wenn nötig
            if idx2 < idx1:
                idx2 += n
            seg = [tuple(contour_pts[j % n]) for j in range(idx1, idx2 + 1)]
            segments.append(seg)

        # Validierung / Fallbacks:
        #    - Sehr kurze Segmente (z.B. < 3 Punkte) oder sehr lange (> 60% der Kontur)
        #      deuten auf Probleme; dann werden die betroffenen Segmente als Gerade
        #      zwischen den jeweiligen Ecken zurückgegeben.
        max_fraction = 0.60
        min_points = 3
        validated_segments = []
        for i, seg in enumerate(segments):
            if len(seg) < min_points or len(seg) > int(n * max_fraction):
                # Fallback: gerade Linie zwischen den geometrischen Ecken
                p1 = tuple(idx_corner_pairs[i][1])
                p2 = tuple(idx_corner_pairs[(i + 1) % 4][1])
                # lineare Interpolation zwischen p1 und p2 (dicht genug)
                x0, y0 = p1
                x1, y1 = p2
                num = max(abs(x1 - x0), abs(y1 - y0)) + 1
                xs = np.linspace(x0, x1, num, dtype=int)
                ys = np.linspace(y0, y1, num, dtype=int)
                linseg = list(zip(xs.tolist(), ys.tolist()))
                validated_segments.append(linseg)
            else:
                validated_segments.append(seg)

        # Klassifiziere die 4 Segmente in top/right/bottom/left basierend auf Segmentmittelpunkt vs Centroid
        cx, cy = self.center_point
        ordered = {"top": [], "right": [], "bottom": [], "left": []}

        for seg in validated_segments:
            if not seg:
                continue
            xs = [p[0] for p in seg]
            ys = [p[1] for p in seg]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            dx = mx - cx
            dy = my - cy
            if abs(dx) > abs(dy):
                if dx > 0:
                    ordered["right"] = seg
                else:
                    ordered["left"] = seg
            else:
                if dy > 0:
                    ordered["bottom"] = seg
                else:
                    ordered["top"] = seg

        return [
            ordered.get("top", []),
            ordered.get("right", []),
            ordered.get("bottom", []),
            ordered.get("left", [])
        ]

    
    def get_edges_from_corners(corners):
        edges = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            edges.append((p1, p2))
        return edges

        #Kanten ansprechen im Vergleichsverfahren
        
        #edges, box = get_puzzle_edges(contour)
        #top_edge = edges[0]     
        #right_edge = edges[1]
        #bottom_edge = edges[2]
        #left_edge = edges[3]

    def get_center_point(self):
        #calculation
        M = cv.moments(self.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            return (0, 0)

    def get_rotated_bounding_box(self):

        rect = cv.minAreaRect(self.contour)   # (center, (width, height), angle)
        box = cv.boxPoints(rect)               # 4 Punkte der Box
        box = np.int32(box)

        # Punkte in Reihenfolge sortieren: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        # Erst y-Koordinate, dann x-Koordinate
        sorted_by_y = sorted(box, key=lambda p: p[1])
        top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])

        tl, tr = top_two
        bl, br = bottom_two

        # Kanten erstellen: jede Kante = [Punkt1, Punkt2]
        top_edge = [tuple(tl), tuple(tr)]
        right_edge = [tuple(tr), tuple(br)]
        bottom_edge = [tuple(br), tuple(bl)]
        left_edge = [tuple(bl), tuple(tl)]

        return [top_edge, right_edge, bottom_edge, left_edge]


    def __repr__(self):
        x, y, w, h = self.bounding_box
        return f"PuzzlePiece {self.index}: Fläche={self.area:.2f}, Box=({x},{y},{w},{h})"
    



    

   
