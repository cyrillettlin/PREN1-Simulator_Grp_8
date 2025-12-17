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

    def get_best_4_corners(self, epsilon_factor=0.00002):
            #Rauschen reduzieren
            epsilon = epsilon_factor * cv.arcLength(self.contour, True)
            approx = cv.approxPolyDP(self.contour, epsilon, True)
            approx_arr = approx.reshape(-1, 2)

            rect = cv.minAreaRect(self.contour)
            box = cv.boxPoints(rect)
            box = np.int32(box)

            real_corners = []

            for box_point in box:
                deltas = approx_arr - box_point
                dists = np.linalg.norm(deltas, axis=1)

                min_idx = np.argmin(dists)

                closest_point = tuple(approx_arr[min_idx])
                real_corners.append(closest_point)


            real_corners = sorted(real_corners, key=lambda p: p[1]) 

            top_group = sorted(real_corners[:2], key=lambda p: p[0]) 
            bottom_group = sorted(real_corners[2:], key=lambda p: p[0], reverse=True) 

            sorted_corners = top_group + bottom_group

            return sorted_corners

    def get_puzzle_edges(self):
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
    



    

   
