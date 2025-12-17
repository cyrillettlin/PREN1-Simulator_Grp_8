import numpy as np
from typing import List, Dict
from edgecomparator import EdgeComparator

class PuzzleMatcher:
#Brute-Force Matcher für Puzzle-Kanten.

    def __init__(self, pieces: List):
        self.pieces = pieces

    def find_matches(self, threshold: float = 0.2) -> List[Dict]:
        matches = []
        for i, pa in enumerate(self.pieces):
            for j, pb in enumerate(self.pieces):
                if i >= j: continue

                for edge_a_idx, edge_a in enumerate(pa.edges):
                    for edge_b_idx, edge_b in enumerate(pb.edges):
                        
                        comp = EdgeComparator(edge_a["points"], edge_b["points"])
                                                
                        score = comp.compare()
                        
                        # Nur hinzufügen, wenn der Score plausibel ist
                        if score < threshold and score < 10.0:
                            matches.append({
                                "piece_a": pa.index,
                                "edge_a": edge_a_idx,
                                "piece_b": pb.index,
                                "edge_b": edge_b_idx,
                                "score": score
                            })
        matches.sort(key=lambda m: m["score"])
        return matches
