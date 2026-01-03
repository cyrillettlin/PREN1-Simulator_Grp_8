class MatchPlacer:
    def __init__(self, ga, rot, ta, logger=None):
        self.ga = ga
        self.rot = rot
        self.ta = ta
        self.log = logger

    def apply_matches(self, matches):
        if not matches:
            if self.log: self.log.warning("Keine Matches gefunden.")
            return

        for m in matches:
            # compute rotation from current geometry
            lines = self.ga.get_matching_edge_lines(m['piece_a'], m['edge_a'], m['piece_b'], m['edge_b'])
            req_rot = self.rot.compute_required_rotation_deg(lines)
        
            pb = self.ga.get_solved_puzzle_piece(m['piece_b'] - 1)
            self.rot.rotate_puzzle_in_place(pb, req_rot)
        
            # IMPORTANT: recompute lines after rotation, so pb edge endpoints are updated
            lines_after = self.ga.get_matching_edge_lines(m['piece_a'], m['edge_a'], m['piece_b'], m['edge_b'])
            self.ta.translate_piece_b_to_a_in_place(pb, lines_after)

