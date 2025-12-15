from puzzle import Puzzle
from edgecomparator import EdgeComparator
from typing import List, Tuple, Dict, Optional


class PuzzleMatcher:
    """
    Findet mögliche Kanten-Matches zwischen mehreren Puzzleteilen.
    """

    def __init__(self, pieces):
        self.pieces = pieces

    def find_matches(self) -> List[Dict]:
        matches = []

        for i, piece_a in enumerate(self.pieces):
            edges_a = piece_a.get_puzzle_edges()

            for j, piece_b in enumerate(self.pieces):
                if i >= j:
                    continue

                edges_b = piece_b.get_puzzle_edges()

                for idx_a, edge_a in enumerate(edges_a):
                    for idx_b, edge_b in enumerate(edges_b):
                        
                        comparator = EdgeComparator(edge_a, edge_b)

                        if comparator.match_edges():
                            matches.append({
                                "piece_a": piece_a.index,
                                "edge_a": idx_a,
                                "piece_b": piece_b.index,
                                "edge_b": idx_b
                            })

        return matches