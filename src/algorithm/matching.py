import sys
import os

import numpy as np
import cv2
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Tuple, Dict, Optional
from PuzzleEdge import *
from puzzle import puzzle


# EdgeComparator
class EdgeComparator:
    """
    Vergleicht Kanten
    """

    logger = logging.getLogger(__name__)
    
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
        logger.debug(f"Contour MSE: {mse:.5f} (tolerance: {tolerance})")

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
        logger.info(f"Edges matched: Length and Contour within tolerances.")
        #TODO: Logging Rückgabe der passenden Puzzleteile, evtl. Kanten.


# PuzzleMatcher
# TODO: entfernen oder in Test verschieben.
class PuzzleMatcher:
    """
    Findet mögliche Kanten-Matches zwischen mehreren Puzzleteilen.
    """

    def __init__(self, pieces: List[puzzle]):
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




