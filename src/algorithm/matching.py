import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from PuzzleEdge import *
from PuzzlePiece import *


# EdgeComparator
class EdgeComparator:
    """
    Vergleicht Kanten
    """

    @staticmethod
    def compare_length(edge_a: PuzzleEdge, edge_b: PuzzleEdge, tolerance: float = 5.0) -> bool:
        diff = abs(edge_a.length - edge_b.length)
        return diff <= tolerance

    @staticmethod
    def compare_contour(edge_a: PuzzleEdge, edge_b: PuzzleEdge, tolerance: float = 0.02) -> bool:
        a = edge_a.normalized_shape
        b = edge_b.normalized_shape
        b_flip = -b[::-1]
        mse = np.mean((a - b_flip) ** 2)
        return mse < tolerance

    @staticmethod
    def match_edges(edge_a: PuzzleEdge, edge_b: PuzzleEdge,
                    length_tolerance: float = 5.0,
                    contour_tolerance: float = 0.02) -> bool:
        """
        1. Länge, dann Kontur
        """
        if not EdgeComparator.compare_length(edge_a, edge_b, length_tolerance):
            return False
        return EdgeComparator.compare_contour(edge_a, edge_b, contour_tolerance)



# PuzzleMatcher

class PuzzleMatcher:
    """
    Findet mögliche Kanten-Matches zwischen mehreren Puzzleteilen.
    """

    def __init__(self, pieces: List[PuzzlePiece]):
        self.pieces = pieces

    def find_matches(self,
                     length_tolerance: float = 5.0,
                     contour_tolerance: float = 0.02) -> List[Dict]:
        """
        Vergleicht alle Kanten aller Teile untereinander.

        Returns
        -------
        List[Dict]
            Liste möglicher Matches mit IDs und Edge-Indices.
        """
        matches = []
        for i, piece_a in enumerate(self.pieces):
            for j, piece_b in enumerate(self.pieces):
                if i >= j:
                    continue  # vermeide doppelte Vergleiche

                for idx_a, edge_a in enumerate(piece_a.edges):
                    for idx_b, edge_b in enumerate(piece_b.edges):
                        if EdgeComparator.match_edges(edge_a, edge_b,
                                                      length_tolerance, contour_tolerance):
                            matches.append({
                                "piece_a": piece_a.id,
                                "edge_a": idx_a,
                                "piece_b": piece_b.id,
                                "edge_b": idx_b
                            })
        return matches



# DummyPuzzleGenerator

class DummyPuzzleGenerator:

    @staticmethod
    def sinus_edge(amplitude: float, phase: float, length: float = 100, points: int = 200) -> PuzzleEdge:
        x = np.linspace(0, length, points)
        y = np.sin((x / length) * 2 * np.pi + phase) * amplitude
        return PuzzleEdge(np.stack([x, y], axis=1))

    @staticmethod
    def create_dummy_piece(piece_id: int) -> PuzzlePiece:
        """
        Erzeugt ein Puzzleteil mit 4 zufälligen Kanten (Sinus-Varianten).
        """
        edges = []
        base_phase = np.random.rand() * 2 * np.pi
        for i in range(4):
            phase = base_phase + i * np.pi / 2
            amp = np.random.uniform(3, 7)
            edges.append(DummyPuzzleGenerator.sinus_edge(amplitude=amp, phase=phase))
        return PuzzlePiece(piece_id, edges)

def mirror_edge_along_axis(edge: PuzzleEdge) -> PuzzleEdge:
    """
    Spiegelt eine Kante entlang ihrer eigenen Achse (Hauptlinie).
    So wird eine geometrisch passende Gegenkante erzeugt.
    """
    points = np.copy(edge.points)

    # Schritt 1: Verschiebe Ursprung zum Startpunkt
    p0 = points[0]
    shifted = points - p0

    # Schritt 2: Rotation, sodass Endpunkt auf x-Achse liegt
    p_end = shifted[-1]
    theta = -np.arctan2(p_end[1], p_end[0])
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = shifted @ R.T

    # Schritt 3: Spiegelung an der x-Achse
    mirrored = rotated.copy()
    mirrored[:, 1] *= -1

    # Schritt 4: Rückrotation & Rückverschiebung
    R_inv = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta),  np.cos(-theta)]
    ])
    restored = mirrored @ R_inv.T
    restored += p0

    return PuzzleEdge(restored)


# Beispiel: Simulation

if __name__ == "__main__":
    np.random.seed(42)  # Reproduzierbarkeit

    piece1 = DummyPuzzleGenerator.create_dummy_piece(1)

    edge_a = piece1.get_edge(0)

    matching_edge = mirror_edge_along_axis(edge_a)
    matching_piece = PuzzlePiece(2, [matching_edge])

    pieces = [piece1, matching_piece]
    matcher = PuzzleMatcher(pieces)
    results = matcher.find_matches(length_tolerance=2.0, contour_tolerance=0.01)

    print("\n=== Gefundene Matches ===")
    if results:
        for match in results:
            print(match)
    else:
        print("❌ Keine passenden Kanten gefunden.")

    from pprint import pprint
    print("\n--- Diagnosedaten ---")
    print(f"Länge Edge 1: {edge_a.length:.3f}")
    print(f"Länge Edge 2: {matching_piece.edges[0].length:.3f}")
    print("Längenvergleich:", EdgeComparator.compare_length(edge_a, matching_piece.edges[0]))
    print("Konturvergleich:", EdgeComparator.compare_contour(edge_a, matching_piece.edges[0]))


import matplotlib.pyplot as plt

plt.plot(edge_a.points[:, 0], edge_a.points[:, 1], label="Original")
plt.plot(matching_edge.points[:, 0], matching_edge.points[:, 1], label="Gespiegelt")
plt.legend()
plt.title("Kante vs. gespiegelte Gegenkante")
plt.show()

