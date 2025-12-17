import matplotlib.pyplot as plt
import numpy as np
from edgecomparator import EdgeComparator

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

        if dist2 < dist1:
            B_plot = B_inv_rev
        else:
            B_plot = B_inv

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(A_res[:, 0], A_res[:, 1], "b-", lw=3, label=f"Teil {pa.index} (Kante {match['edge_a']})")
        ax.plot(B_plot[:, 0], B_plot[:, 1], "r--", lw=3, label=f"Teil {pb.index} (Kante {match['edge_b']}, gespiegelt)")
        
        ax.set_title(f"Match {i+1}: Score = {match['score']:.4f}")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(-0.6, 0.6) 
        plt.show()