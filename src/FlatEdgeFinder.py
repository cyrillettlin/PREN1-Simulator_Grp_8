import logging
from typing import List, Tuple, Dict, Optional
from edgecomparator import EdgeComparator


class FlatEdgeFinder:
    """
    Utility class for edge classification / anchor preparation.

    Responsibilities (for now):
      - classify a single edge as flat/tab/hole
      - list edge types for a piece
      - extract flat edges for a piece
      - optionally log edge types for a list of pieces
    """

    def __init__(self, num_points: int = 100, logger: Optional[logging.Logger] = None):
        self.num_points = int(num_points)
        self.log = logger or logging.getLogger(__name__)

    def classify_edge_points(self, edge_points) -> str:
        import numpy as np
        from edgecomparator import EdgeComparator

        arr = np.asarray(edge_points, dtype=np.float32)

        try:
            arr = arr.reshape(-1, 2)
        except Exception:
            self.log.warning(f"Edge points not reshapeable to Nx2. shape={arr.shape}.")
            return "unknown"

        # Need at least 2 points to define an edge
        if arr.shape[0] < 2:
            return "unknown"

        comp = EdgeComparator(arr, arr, num_points=self.num_points)

        norm = comp._normalize_geometry(comp.edge_a)
        norm = np.asarray(norm, dtype=np.float32).reshape(-1, 2)
        if norm.shape[0] < 2:
            return "unknown"

        res = comp._resample_edge(norm)
        res = np.asarray(res, dtype=np.float32).reshape(-1, 2)
        if res.shape[0] < 2:
            return "unknown"

        return comp.get_edge_type(res)



    def edge_types(self, piece) -> List[Tuple[int, str]]:
        """
        Return list of (edge_index, edge_type) for the given piece.
        """
        return [(i, self.classify_edge_points(e["points"])) for i, e in enumerate(piece.get_puzzle_edges())]

    def flat_edges(self, piece) -> List[int]:
        """
        Return only the indices of edges classified as 'flat'.
        """
        types = self.edge_types(piece)
        logging.info(f"Edge_types: {types}")
        for i, t in types:
            if t == 'flat':
                logging.info(f"Flat Edge: {i,t}")
        logging.info(f"")
        return [i for i, t in types if t == 'flat']

    def log_edge_types(self, pieces) -> None:
        """
        Convenience: log edge types for every piece in pieces.
        """
        for p in pieces:
            self.log.info(f"Piece {p.index} edge types: {self.edge_types(p)}")