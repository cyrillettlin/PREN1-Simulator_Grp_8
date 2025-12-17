import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from edgecomparator import EdgeComparator

class Visualizer:

    @staticmethod
    def show_matches(matches, pieces):
        piece_map = {p.index: p for p in pieces}

        for i, match in enumerate(matches):
            pa = piece_map[match["piece_a"]]
            pb = piece_map[match["piece_b"]]

            edge_a = pa.edges[match["edge_a"]]["points"]
            edge_b = pb.edges[match["edge_b"]]["points"]

            comp = EdgeComparator(edge_a, edge_b)
            
            A_norm = comp._normalize_geometry(np.array(edge_a))
            B_norm = comp._normalize_geometry(np.array(edge_b))
            
            A_res = comp._resample_edge(A_norm)
            B_res = comp._resample_edge(B_norm)

            B_inv = B_res.copy()
            B_inv[:, 1] *= -1
            
            dist1 = np.linalg.norm(A_res - B_inv)

            B_inv_rev = B_inv[::-1].copy()
            B_inv_rev[:, 0] = 1.0 - B_inv_rev[:, 0]
            dist2 = np.linalg.norm(A_res - B_inv_rev)

            B_plot = B_inv_rev if dist2 < dist1 else B_inv

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(A_res[:, 0], A_res[:, 1], "b-", lw=3, label=f"Teil {pa.index} (Kante {match['edge_a']})")
            ax.plot(B_plot[:, 0], B_plot[:, 1], "r--", lw=3, label=f"Teil {pb.index} (Kante {match['edge_b']}, gespiegelt)")
            
            ax.set_title(f"Match {i+1}: Score = {match['score']:.4f}")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True)
            ax.set_ylim(-0.6, 0.6) 
            plt.show()

    @staticmethod
    def show_all_edges(pieces, image=None):
        #Zeigt jedes Puzzle-Teil mit Kanten, Ecken und Nummer in der Mitte.

        colors = [
            (0, 255, 255),  # top
            (255, 0, 255),  # right
            (255, 255, 0),  # bottom
            (0, 165, 255)   # left
        ]

        for piece in pieces:
            x, y, w, h = piece.bounding_box
            canvas = np.zeros((h + 20, w + 20, 3), dtype=np.uint8)
            if image is not None:
                canvas[10:10+h, 10:10+w] = image[y:y+h, x:x+w]

            edges = piece.get_puzzle_edges()
            for i, edge in enumerate(edges):
                pts = edge.get("points", [])
                if pts:
                    pts_rel = [(p[0]-x+10, p[1]-y+10) for p in pts]
                    cv.polylines(canvas, [np.array(pts_rel, dtype=np.int32)], isClosed=False, color=colors[i % 4], thickness=2)

            # Ecken einzeichnen
            corners = piece.get_best_4_corners()
            for j, c in enumerate(corners):
                cx, cy = c
                cv.circle(canvas, (cx - x + 10, cy - y + 10), 5, (0, 0, 255), -1)
                cv.putText(canvas, f"{j+1}", (cx - x + 12, cy - y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Nummer in der Mitte
            cx, cy = piece.center_point
            cv.circle(canvas, (cx - x + 10, cy - y + 10), 7, (0, 255, 0), -1)
            cv.putText(canvas, str(piece.index), (cx - x + 3, cy - y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv.imshow(f"Puzzle {piece.index} Edges", canvas)

        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def show_all_edges_grid(pieces, image=None, padding=20):
        """
        Zeigt alle Puzzle-Teile in einem Raster mit Kanten, Ecken und Nummern wie show_all_edges.
        """
        colors = [
            (0, 255, 255),  # top
            (255, 0, 255),  # right
            (255, 255, 0),  # bottom
            (0, 165, 255)   # left
        ]

        n = len(pieces)
        cols = math.ceil(np.sqrt(n))
        rows = math.ceil(n / cols)

        max_w = max([p.bounding_box[2] for p in pieces]) + padding
        max_h = max([p.bounding_box[3] for p in pieces]) + padding

        canvas = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

        for idx, piece in enumerate(pieces):
            row = idx // cols
            col = idx % cols
            x_offset = col * max_w
            y_offset = row * max_h

            x, y, w, h = piece.bounding_box
            tile = np.zeros((h + 20, w + 20, 3), dtype=np.uint8)
            if image is not None:
                tile[10:10+h, 10:10+w] = image[y:y+h, x:x+w]

            edges = piece.get_puzzle_edges()
            for i, edge in enumerate(edges):
                pts = edge.get("points", [])
                if pts:
                    pts_rel = [(p[0]-x+10, p[1]-y+10) for p in pts]
                    cv.polylines(tile, [np.array(pts_rel, dtype=np.int32)], isClosed=False, color=colors[i % 4], thickness=2)

            corners = piece.get_best_4_corners()
            for j, c in enumerate(corners):
                cx, cy = c
                cv.circle(tile, (cx - x + 10, cy - y + 10), 5, (0, 0, 255), -1)
                cv.putText(tile, f"{j+1}", (cx - x + 12, cy - y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Nummer in der Mitte
            cx, cy = piece.center_point
            cv.circle(tile, (cx - x + 10, cy - y + 10), 7, (0, 255, 0), -1)
            cv.putText(tile, str(piece.index), (cx - x + 3, cy - y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            th, tw = tile.shape[:2]
            canvas[y_offset:y_offset+th, x_offset:x_offset+tw] = tile

        cv.imshow("Alle Puzzleteile mit erkannten Konturen und Ecken", canvas)
        cv.waitKey(0)
        cv.destroyAllWindows()
