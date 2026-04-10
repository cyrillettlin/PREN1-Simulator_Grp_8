import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Puzzle:

    # ── Schwellenwerte für die BB-Zuverlässigkeitsprüfung ──────────────────

    # Nase-Zentralität: 0 = Rand, 1 = perfekte Mitte der Seite.
    BB_CENTRALITY_THRESHOLD = 0.60

    # Normierte Distanz Nasenspitze → nächste BB-Ecke (/ BB-Diagonale).
    BB_NOSE_CORNER_DIST_THR = 0.20

    # Mindest-Amplitude einer Nase (normiert), damit sie bewertet wird.
    BB_MIN_NOSE_AMPLITUDE = 0.03

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

    def set_contour(self, cnt):
        self.contour = cnt

    # ══════════════════════════════════════════════════════════════════════
    # Öffentliche Hauptmethode
    # ══════════════════════════════════════════════════════════════════════

    def get_best_4_corners(self, epsilon_factor=0.00002, cnn_predict_fn=None):
        """
        Wählt automatisch zwischen Bounding-Box- und CNN-Ansatz.

        Checks (in Reihenfolge):
          1. Nase-Zentralität  – sitzt eine Nase zu mittig auf einer Seite?
          2. Nase-Ecken-Distanz – liegt eine Nasenspitze zu nahe an BB-Ecke?
             (erkennt konvexe Nasen UND konkave Slots)

        Schlägt ein Check an → CNN (wenn verfügbar), sonst BB mit Warning.
        """
        reliable, reason = self._bb_is_reliable()

        if not reliable:
            logger.info(
                f"Teil {self.index}: BB nicht zuverlaessig ({reason}) "
                f"-> {'CNN' if cnn_predict_fn else 'BB trotzdem (kein CNN)'}"
            )
            if cnn_predict_fn is not None:
                return cnn_predict_fn()
            logger.warning(
                f"Teil {self.index}: Kein CNN verfuegbar, BB wird trotzdem verwendet."
            )
        else:
            logger.info(f"Teil {self.index}: BB zuverlaessig ({reason})")

        return self._get_corners_bounding_box(epsilon_factor)

    # ══════════════════════════════════════════════════════════════════════
    # Check-Logik
    # ══════════════════════════════════════════════════════════════════════

    def _bb_is_reliable(self):
        """
        Prüft ob der BB-Ansatz zuverlässige Ecken liefert.
        Checks pro Seite: Nasen-Zentralität und Nasen-Ecken-Distanz.
        """
        pts = self.contour.reshape(-1, 2).astype(float)
        x, y, w, h = self.bounding_box
        bb_diag = float(np.hypot(w, h))

        bb_corners = np.array([
            [x,     y    ],   # OL
            [x + w, y    ],   # OR
            [x,     y + h],   # UL
            [x + w, y + h],   # UR
        ], dtype=float)

        side_configs = [
            # (name, Längsachse, Senkrechtachse, Richtung, Seitenstart, Seitenlänge, Ecken-Indizes)
            ("oben",   0, 1, "min", x, w, [0, 1]),
            ("unten",  0, 1, "max", x, w, [2, 3]),
            ("links",  1, 0, "min", y, h, [0, 2]),
            ("rechts", 1, 0, "max", y, h, [1, 3]),
        ]

        for name, ax, perp, direction, start, length, corner_idxs in side_configs:
            result = self._analyze_side(
                pts, ax, perp, direction, start, length,
                bb_corners, corner_idxs, bb_diag
            )
            if result is None:
                continue

            centrality, nose_corner_dist, nose_type = result

            logger.debug(
                f"  Seite '{name}' [{nose_type}]: "
                f"Zentralitaet={centrality:.2f}, BB-Dist={nose_corner_dist:.2f}"
            )

            if centrality > self.BB_CENTRALITY_THRESHOLD:
                return False, (
                    f"Seite '{name}' [{nose_type}]: "
                    f"Nase-Zentralitaet {centrality:.2f} > {self.BB_CENTRALITY_THRESHOLD}"
                )

            if nose_corner_dist < self.BB_NOSE_CORNER_DIST_THR:
                return False, (
                    f"Seite '{name}' [{nose_type}]: "
                    f"Nase-Ecken-Distanz {nose_corner_dist:.2f} < {self.BB_NOSE_CORNER_DIST_THR}"
                )

        return True, "Alle Checks bestanden"

    def _analyze_side(self, pts, ax, perp, direction, start, length,
                      bb_corners, corner_idxs, bb_diag):
        """
        Analysiert eine Seite auf gefährliche Nasenposition.

        Erkennt BEIDE Typen:
          - Konvexe Nase: ragt über die BB-Kante hinaus (wie bei Bild 1)
          - Konkaver Slot: Einbuchtung ins Innere (wie bei Bild 2/3)

        Der dominante Typ (grösste Amplitude) wird bewertet.
        """
        margin = 0.08 * length
        mask = (
            (pts[:, ax] >= start + margin) &
            (pts[:, ax] <= start + length - margin)
        )
        side_pts = pts[mask]
        if len(side_pts) < 5:
            return None

        if direction == "min":
            bb_edge_val = bb_corners[corner_idxs[0], perp]
        else:
            bb_edge_val = bb_corners[corner_idxs[1], perp]

        perp_vals = side_pts[:, perp]

        if direction == "min":
            # Konvex: ragt über die BB nach aussen (perp < bb_edge)
            convex_amp  = max(0.0, bb_edge_val - perp_vals.min())
            # Slot: Einbuchtung nach innen (perp > bb_edge)
            concave_amp = max(0.0, perp_vals.max() - bb_edge_val)
            if convex_amp >= concave_amp:
                nose_type = "konvex"
                tip_idx = int(np.argmin(perp_vals))
            else:
                nose_type = "konkav/slot"
                tip_idx = int(np.argmax(perp_vals))
        else:
            convex_amp  = max(0.0, perp_vals.max() - bb_edge_val)
            concave_amp = max(0.0, bb_edge_val - perp_vals.min())
            if convex_amp >= concave_amp:
                nose_type = "konvex"
                tip_idx = int(np.argmax(perp_vals))
            else:
                nose_type = "konkav/slot"
                tip_idx = int(np.argmin(perp_vals))

        amplitude = max(convex_amp, concave_amp) / bb_diag
        if amplitude < self.BB_MIN_NOSE_AMPLITUDE:
            return None

        nose_tip = side_pts[tip_idx]
        rel = (nose_tip[ax] - start) / length
        centrality = 1.0 - 2.0 * abs(rel - 0.5)

        dists = [np.linalg.norm(nose_tip - bb_corners[i]) for i in corner_idxs]
        nose_corner_dist = min(dists) / bb_diag

        return centrality, nose_corner_dist, nose_type

    # ══════════════════════════════════════════════════════════════════════
    # BB-Ecken-Berechnung (originale Logik, unverändert)
    # ══════════════════════════════════════════════════════════════════════

    def _get_corners_bounding_box(self, epsilon_factor=0.00002):
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
            real_corners.append(tuple(approx_arr[min_idx]))

        real_corners = sorted(real_corners, key=lambda p: p[1])
        top_group    = sorted(real_corners[:2], key=lambda p: p[0])
        bottom_group = sorted(real_corners[2:], key=lambda p: p[0], reverse=True)

        return top_group + bottom_group

    # ══════════════════════════════════════════════════════════════════════
    # Rest der Klasse (unverändert)
    # ══════════════════════════════════════════════════════════════════════

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

        edges = [
            {"points": ordered.get("top", []),    "type": "inner"},
            {"points": ordered.get("right", []),  "type": "inner"},
            {"points": ordered.get("bottom", []), "type": "inner"},
            {"points": ordered.get("left", []),   "type": "inner"},
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

    def get_rotated_bounding_box(self):
        rect = cv.minAreaRect(self.contour)
        box = cv.boxPoints(rect)
        box = np.int32(box)
        sorted_by_y = sorted(box, key=lambda p: p[1])
        top_two    = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
        tl, tr = top_two
        bl, br = bottom_two
        return [
            [tuple(tl), tuple(tr)],
            [tuple(tr), tuple(br)],
            [tuple(br), tuple(bl)],
            [tuple(bl), tuple(tl)],
        ]

    def __repr__(self):
        x, y, w, h = self.bounding_box
        return f"PuzzlePiece {self.index}: Flaeche={self.area:.2f}, Box=({x},{y},{w},{h})"