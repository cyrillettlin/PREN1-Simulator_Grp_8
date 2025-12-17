from collections import deque

class PuzzleOrganizer:
    EDGE_DIR = {
        0: "top",
        1: "right",
        2: "bottom",
        3: "left"
    }

    OPPOSITE = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left"
    }

    OFFSETS = {
        "top": (-1, 0),
        "right": (0, 1),
        "bottom": (1, 0),
        "left": (0, -1)
    }

    def __init__(self, pieces, matches, grid_size=2):
        self.pieces = pieces
        self.matches = matches
        self.grid_size = grid_size
        self.positions = {}
        self._matches_by_piece = self._group_matches()

    def _group_matches(self):
        grouped = {}
        for m in self.matches:
            grouped.setdefault(m["piece_a"], []).append(m)
            grouped.setdefault(m["piece_b"], []).append(m)
        return grouped

    def _build_positions(self):
        if not self.matches:
            return

        start = self.matches[0]["piece_a"]
        self.positions = {start: (0, 0)}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            curr_r, curr_c = self.positions[current]

            for m in self._matches_by_piece.get(current, []):
                if m["piece_a"] == current:
                    other = m["piece_b"]
                    c_edge_idx = m["edge_a"]
                    o_edge_idx = m["edge_b"]
                else:
                    other = m["piece_a"]
                    c_edge_idx = m["edge_b"]
                    o_edge_idx = m["edge_a"]

                if other in self.positions:
                    continue

                c_edge = self.EDGE_DIR[c_edge_idx]
                o_edge = self.EDGE_DIR[o_edge_idx]

                if self.OPPOSITE[c_edge] != o_edge:
                    continue

                dr, dc = self.OFFSETS[c_edge]
                new_pos = (curr_r + dr, curr_c + dc)

                if new_pos in self.positions.values():
                    continue

                self.positions[other] = new_pos
                queue.append(other)

    def organize(self):
        self._build_positions()

        if not self.positions:
            return [[None] * self.grid_size for _ in range(self.grid_size)]

        all_r = [r for r, _ in self.positions.values()]
        all_c = [c for _, c in self.positions.values()]

        min_r, max_r = min(all_r), max(all_r)
        min_c, max_c = min(all_c), max(all_c)

        rows = max(self.grid_size, max_r - min_r + 1)
        cols = max(self.grid_size, max_c - min_c + 1)

        grid = [[None for _ in range(cols)] for _ in range(rows)]

        for p_idx, (r, c) in self.positions.items():
            grid[r - min_r][c - min_c] = p_idx

        # Letztes Puzzlestück einsetzen
        all_indices = {p.index for p in self.pieces}
        placed = set(self.positions.keys())
        missing = list(all_indices - placed)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] is None and missing:
                    grid[r][c] = missing.pop(0)

        return [row[:self.grid_size] for row in grid[:self.grid_size]]
