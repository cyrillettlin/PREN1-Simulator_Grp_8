from PuzzleEdge import *
from typing import List

class PuzzlePiece:

    def __init__(self, piece_id: int, edges: List[PuzzleEdge]):
        self.id = piece_id
        self.edges = edges

    def get_edge(self, index: int) -> PuzzleEdge:
        return self.edges[index]

    def __repr__(self):
        return f"PuzzlePiece(id={self.id}, edges={len(self.edges)})"
    

