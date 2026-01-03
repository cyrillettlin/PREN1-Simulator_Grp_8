from __future__ import annotations
import logging
from typing import Tuple, List, Optional

import numpy as np
class Anchor:
    def __init__(self, flat_finder, rot, ta, logger: Optional[logging.Logger] = None):
        self.flat_finder = flat_finder
        self.rot = rot
        self.ta = ta
        self.log = logger or logging.getLogger(__name__)
        self.EDGE_TO = ((0, 1), (1, 2), (2, 3), (3, 0))

    def choose_anchor(self, solved_puzzles):
        """
        Pick an anchor piece and return (anchor_piece, flat_edges).
        Current policy:
          - use first piece, but warn if it doesn't have exactly 2 flat edges
        """
        anchor = solved_puzzles[0]
        flats = self.flat_finder.flat_edges(anchor)

        if len(flats) != 2:
            self.log.warning(
                f"Anchor piece {getattr(anchor, 'index', '?')} has {len(flats)} flat edges: {flats}. "
                "Corner anchoring expects exactly 2."
            )
        return anchor, flats

    def _shared_corner_idx(self,edge_a: int, edge_b: int) -> int:
        common = list(set(self.EDGE_TO[edge_a % 4]) & set(self.EDGE_TO[edge_b % 4]))
        if len(common) != 1:
            raise ValueError(f"Edges {edge_a} and {edge_b} do not share exactly one corner")
        return common[0]

    def place_anchor(self, anchor_piece, flat_edges, dir1, dir2, target_corner_point):
        ang = self.rot.anchor_rotation_for_corner_deg(anchor_piece, flat_edges, dir1, dir2)
        self.rot.rotate_puzzle_in_place(anchor_piece, ang)

        # IMPORTANT: snap the shared corner of the two flat edges
        k = self._shared_corner_idx(flat_edges[0], flat_edges[1])
        src = np.asarray(anchor_piece.get_best_4_corners(), dtype=float).reshape(4, 2)[k]


        dxdy = self.ta.delta_xy(src, target_corner_point)
        self.ta.translate_puzzle_in_place(anchor_piece, dxdy)

        return ang, dxdy
