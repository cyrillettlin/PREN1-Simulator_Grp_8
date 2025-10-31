from PuzzlePiece import PuzzlePiece
from PuzzleEdge import PuzzleEdge

import numpy as np
import matplotlib.pyplot as plt
from PuzzleEdge import PuzzleEdge
from PuzzlePiece import PuzzlePiece

# Koordinaten eines Puzzelstückes für piece 1
x = [2, 6, 6, 7, 7, 9, 9, 8, 8, 9, 9, 6, 6, 5, 5, 2]
y = [2, 2, 1, 1, 2, 2, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6]
points = np.column_stack((x, y))

# Kanten piece 1
edges_points = [
    points[0:6],   # untere Kante
    points[5:11],   # rechte Kante
    points[10:16],  # obere Kante
    np.vstack((points[15:], points[0]))  # linke Kante
]

edges1 = [PuzzleEdge(pts) for pts in edges_points]
piece1 = PuzzlePiece(piece_id=1, edges=edges1)

# Plot
plt.figure(figsize=(6, 6))
colors = ["red", "green", "blue", "orange"]

# Piece 1
for i, edge in enumerate(piece1.edges):
    pts = edge.points
    plt.plot(pts[:, 0], pts[:, 1], color=colors[i], linewidth=2, label=f"Edge {i+1}")
    plt.scatter(pts[:, 0], pts[:, 1], color=colors[i], s=25)

# Beschriftung und Darstellung
plt.title(f"PuzzlePiece ID={piece1.id} mit 4 Kanten")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


