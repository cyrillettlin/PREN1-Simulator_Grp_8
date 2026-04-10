import cv2 as cv
import numpy as np

class Puzzle:

    def __init__(self, contour, index):
        self.index = index
        self.contour = contour
        self.area = cv.contourArea(contour)
        self.bounding_box = cv.boundingRect(contour)
        self.center_point = self.get_center_point()
        self.edges = []
        self.corners = []


    def get_contour(self):
        return self.contour

    def set_contour(self,cnt):
        self.contour =cnt

    @staticmethod
    def get_angle(p1, p2, p3):
        """Berechnet den Winkel bei p2 in Grad"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 180

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    @staticmethod
    def is_convex(p1, p2, p3):
        """prüft, ob Ecke p2 nach aussen geht"""
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        # Kreuzprodukt
        res = v1[0] * v2[1] - v1[1] * v2[0]
        # Wechseln, jeh nach dem ob Speicherung im Uhrzeigersinn oder Gegenuhrzeigersinn
        return res < 0

    def get_best_4_corners(self, epsilon_factor=0.04): #epsilon_factor=0.00002
            #Rauschen reduzieren
            epsilon = epsilon_factor * cv.arcLength(self.contour, True)
            approx = cv.approxPolyDP(self.contour, epsilon, True)
            approx_arr = approx.reshape(-1, 2)

            n = len(approx_arr)
            cx, cy = self.center_point

            # Nur Punkte behalten, die ca. 90 Grad haben und nach aussen zeigen
            filtered_points = []
            for i in range(n):
                p_prev = approx_arr[i - 1]
                p_curr = approx_arr[i]
                p_next = approx_arr[(i + 1) % n]

                angle = self.get_angle(p_prev, p_curr, p_next)

                if 70 <= angle <= 115:
                    if self.is_convex(p_prev, p_curr, p_next):
                        filtered_points.append(p_curr)

            # Falls keine Winkel-> alle Punkte
            if len(filtered_points) < 4:
                search_pool = approx_arr
            else:
                #Distanz zum Zentrum nutzen
                candidates = sorted(filtered_points,
                                    key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
                                    reverse=True)
                search_pool = np.array(candidates)



            rect = cv.minAreaRect(self.contour)
            box = cv.boxPoints(rect)
            box = np.int32(box)

            real_corners = []

            for box_point in box:
                deltas = search_pool - box_point
                dists = np.linalg.norm(deltas, axis=1)

                min_idx = np.argmin(dists)

                real_corners.append(tuple(search_pool[min_idx]))


            real_corners = sorted(real_corners, key=lambda p: p[1])

            top_group = sorted(real_corners[:2], key=lambda p: p[0])
            bottom_group = sorted(real_corners[2:], key=lambda p: p[0], reverse=True)

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

        if n == 0 or len(corners) != 4:
            edges = [{"points": [], "type": "inner"} for _ in range(4)]
            self.edges = edges
            return edges

        corner_candidate_indices = []
        for c in corners:
            dists = np.linalg.norm(contour_pts - np.array(c), axis=1)
            sorted_idx = np.argsort(dists)
            corner_candidate_indices.append(list(sorted_idx))

        used = set()
        assigned = [None] * 4
        for i in range(4):
            for idx in corner_candidate_indices[i]:
                if idx not in used:
                    assigned[i] = int(idx)
                    used.add(idx)
                    break
            if assigned[i] is None:
                base = corner_candidate_indices[i][0]
                found = False
                for offset in range(1, n):
                    for cand in [(base + offset) % n, (base - offset) % n]:
                        if cand not in used:
                            assigned[i] = int(cand)
                            used.add(cand)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    assigned[i] = int(base)

        idx_corner_pairs = list(zip(assigned, corners))
        idx_corner_pairs.sort(key=lambda x: x[0])

        segments = []
        for i in range(4):
            idx1 = idx_corner_pairs[i][0]
            idx2 = idx_corner_pairs[(i + 1) % 4][0]
            if idx2 < idx1:
                idx2 += n
            seg = [tuple(contour_pts[j % n]) for j in range(idx1, idx2 + 1)]
            segments.append(seg)

        # Validierung: kurze oder lange Segmente ersetzen
        max_fraction = 0.90
        min_points = 3
        validated_segments = []
        for i, seg in enumerate(segments):
            if len(seg) < min_points or len(seg) > int(n * max_fraction):
                p1 = tuple(idx_corner_pairs[i][1])
                p2 = tuple(idx_corner_pairs[(i + 1) % 4][1])
                num = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1
                xs = np.linspace(p1[0], p2[0], num, dtype=int)
                ys = np.linspace(p1[1], p2[1], num, dtype=int)
                validated_segments.append(list(zip(xs.tolist(), ys.tolist())))
            else:
                validated_segments.append(seg)

        # Klassifizierung top/right/bottom/left basierend auf Mittelpunkt
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

        # Kanten als Dict mit Typ zurückgeben
        edges = [
            {"points": ordered.get("top", []), "type": "inner"},
            {"points": ordered.get("right", []), "type": "inner"},
            {"points": ordered.get("bottom", []), "type": "inner"},
            {"points": ordered.get("left", []), "type": "inner"},
        ]

        self.edges = edges
        return edges


    def get_center_point(self):
        M = cv.moments(self.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            return (0, 0)

    #Aktuell nicht verwendet, aber für Rotationtest notwendig
    def get_rotated_bounding_box(self):

        rect = cv.minAreaRect(self.contour)
        box = cv.boxPoints(rect)
        box = np.int32(box)

        sorted_by_y = sorted(box, key=lambda p: p[1])
        top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])

        tl, tr = top_two
        bl, br = bottom_two

        top_edge = [tuple(tl), tuple(tr)]
        right_edge = [tuple(tr), tuple(br)]
        bottom_edge = [tuple(br), tuple(bl)]
        left_edge = [tuple(bl), tuple(tl)]

        return [top_edge, right_edge, bottom_edge, left_edge]

    def __repr__(self):
        x, y, w, h = self.bounding_box
        return f"PuzzlePiece {self.index}: Fläche={self.area:.2f}, Box=({x},{y},{w},{h})"
    



    

   
