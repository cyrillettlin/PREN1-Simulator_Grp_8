import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from edgecomparator import EdgeComparator


class Visualizer:

    # =========================
    # 🔍 ZOOM FEATURE
    # =========================
    @staticmethod
    def _interactive_zoom(window_name, image):
        zoom = 1.0
        min_zoom, max_zoom = 0.2, 5.0

        def redraw():
            resized = cv.resize(image, None, fx=zoom, fy=zoom, interpolation=cv.INTER_LINEAR)
            cv.imshow(window_name, resized)

        def mouse_callback(event, x, y, flags, param):
            nonlocal zoom

            if event == cv.EVENT_MOUSEWHEEL:
                if flags > 0:
                    zoom *= 1.1
                else:
                    zoom /= 1.1

                zoom = max(min_zoom, min(max_zoom, zoom))
                redraw()

        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, mouse_callback)

        redraw()

        while True:
            key = cv.waitKey(20)
            if key == 27:  # ESC
                break

        cv.destroyWindow(window_name)

    # =========================
    # DRAW HELPERS
    # =========================
    @staticmethod
    def _draw_dashed_rect(img, pts, color, thickness=1, dash=8, gap=6):
        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = np.hypot(dx, dy)
            if dist == 0:
                continue
            ux, uy = dx / dist, dy / dist
            pos, drawing = 0.0, True
            while pos < dist:
                seg_len = dash if drawing else gap
                end = min(pos + seg_len, dist)
                if drawing:
                    sp = (int(p1[0] + ux * pos), int(p1[1] + uy * pos))
                    ep = (int(p1[0] + ux * end), int(p1[1] + uy * end))
                    cv.line(img, sp, ep, color, thickness)
                pos = end
                drawing = not drawing

    @staticmethod
    def _draw_bounding_box(canvas, piece, x_off=10, y_off=10):
        bx, by, bw, bh = piece.bounding_box

        pts = piece.contour.reshape(-1, 2).astype(np.float32)
        shift = np.array([x_off - bx, y_off - by], dtype=np.float32)
        pts_rel = pts + shift

        # Axis-Aligned BB
        aa_bb = np.array([
            [x_off, y_off],
            [x_off + bw, y_off],
            [x_off + bw, y_off + bh],
            [x_off, y_off + bh],
        ], dtype=np.int32)

        Visualizer._draw_dashed_rect(canvas, aa_bb, color=(200, 200, 200), thickness=1)
        cv.putText(canvas, "AABB", (x_off + 3, y_off + 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        # Rotated BB
        rect = cv.minAreaRect(pts_rel.reshape(-1, 1, 2))
        box = cv.boxPoints(rect).astype(np.int32)

        cv.polylines(canvas, [box.reshape(-1, 1, 2)], True, (0, 220, 220), 1)

        for k, bpt in enumerate(box):
            cv.drawMarker(canvas, tuple(bpt), (0, 220, 220), cv.MARKER_DIAMOND, 8, 1)
            cv.putText(canvas, f"R{k+1}", (bpt[0] + 4, bpt[1] - 3),
                       cv.FONT_HERSHEY_SIMPLEX, 0.28, (0, 220, 220), 1)

    # =========================
    # MATCH VISUALIZATION
    # =========================
    @staticmethod
    def show_matches(matches, pieces):
        piece_map = {p.index: p for p in pieces}

        for i, match in enumerate(matches):
            pa = piece_map[match["piece_a"]]
            pb = piece_map[match["piece_b"]]

            edge_a = pa.edges[match["edge_a"]]["points"]
            edge_b = pb.edges[match["edge_b"]]["points"]

            comp = EdgeComparator(edge_a, edge_b)

            A = comp._resample_edge(comp._normalize_geometry(np.array(edge_a)))
            B = comp._resample_edge(comp._normalize_geometry(np.array(edge_b)))

            B[:, 1] *= -1
            B_rev = B[::-1].copy()
            B_rev[:, 0] = 1.0 - B_rev[:, 0]

            B_plot = B_rev if np.linalg.norm(A - B_rev) < np.linalg.norm(A - B) else B

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(A[:, 0], A[:, 1], "b-", lw=3, label=f"Teil {pa.index}")
            ax.plot(B_plot[:, 0], B_plot[:, 1], "r--", lw=3, label=f"Teil {pb.index}")

            ax.set_title(f"Match {i+1}: Score = {match['score']:.4f}")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True)
            ax.set_ylim(-0.6, 0.6)
            plt.show()

    # =========================
    # SINGLE VIEW
    # =========================
    @staticmethod
    def show_all_edges(pieces, image=None):
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]

        for piece in pieces:
            x, y, w, h = piece.bounding_box
            canvas = np.zeros((h + 20, w + 20, 3), dtype=np.uint8)

            if image is not None:
                canvas[10:10+h, 10:10+w] = image[y:y+h, x:x+w]

            # Edges
            for i, edge in enumerate(piece.get_puzzle_edges()):
                pts = edge.get("points", [])
                if pts:
                    pts_rel = [(p[0]-x+10, p[1]-y+10) for p in pts]
                    cv.polylines(canvas, [np.array(pts_rel, np.int32)], False, colors[i % 4], 2)

            Visualizer._draw_bounding_box(canvas, piece)

            # Corners
            for j, (cx, cy) in enumerate(piece.get_best_4_corners()):
                cv.circle(canvas, (cx-x+10, cy-y+10), 5, (0, 0, 255), -1)
                cv.putText(canvas, str(j+1), (cx-x+13, cy-y+5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Center
            pcx, pcy = piece.center_point
            cv.circle(canvas, (pcx-x+10, pcy-y+10), 7, (0, 255, 0), -1)

            Visualizer._interactive_zoom(f"Puzzle {piece.index}", canvas)

    # =========================
    # GRID VIEW
    # =========================
    @staticmethod
    def show_all_edges_grid(pieces, image=None, padding=20):
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]

        n = len(pieces)
        cols = math.ceil(np.sqrt(n))
        rows = math.ceil(n / cols)

        max_w = max(p.bounding_box[2] for p in pieces) + padding
        max_h = max(p.bounding_box[3] for p in pieces) + padding

        canvas = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

        for idx, piece in enumerate(pieces):
            row, col = divmod(idx, cols)
            x_off, y_off = col * max_w, row * max_h

            x, y, w, h = piece.bounding_box
            tile = np.zeros((h + 20, w + 20, 3), dtype=np.uint8)

            if image is not None:
                tile[10:10+h, 10:10+w] = image[y:y+h, x:x+w]

            for i, edge in enumerate(piece.get_puzzle_edges()):
                pts = edge.get("points", [])
                if pts:
                    pts_rel = [(p[0]-x+10, p[1]-y+10) for p in pts]
                    cv.polylines(tile, [np.array(pts_rel, np.int32)], False, colors[i % 4], 2)

            Visualizer._draw_bounding_box(tile, piece)

            for j, (cx, cy) in enumerate(piece.get_best_4_corners()):
                cv.circle(tile, (cx-x+10, cy-y+10), 5, (0, 0, 255), -1)
                cv.putText(tile, str(j+1), (cx-x+13, cy-y+5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            pcx, pcy = piece.center_point
            cv.circle(tile, (pcx-x+10, pcy-y+10), 7, (0, 255, 0), -1)

            th, tw = tile.shape[:2]
            canvas[y_off:y_off+th, x_off:x_off+tw] = tile

        Visualizer._interactive_zoom("Alle Puzzleteile", canvas)