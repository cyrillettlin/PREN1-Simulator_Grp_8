import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from matching import Matching

class MockPiece:
    def __init__(self, index, edges):
        self.index = index
        self.edges = [{"points": e} for e in edges]

class TestPuzzleMatcher(unittest.TestCase):

    def setUp(self):
        # Einfach definierte Kanten (2D Punkte)
        edge_tab = np.array([[0, 0], [0.5, 0.2], [1, 0]])
        edge_hole = np.array([[0, 0], [0.5, -0.2], [1, 0]])
        edge_flat = np.array([[0, 0], [0.5, 0], [1, 0]])

        # Zwei Pieces, jeweils mit mehreren Kanten
        self.piece1 = MockPiece(0, [edge_tab, edge_flat])
        self.piece2 = MockPiece(1, [edge_hole, edge_flat])
        self.piece3 = MockPiece(2, [edge_tab, edge_hole])

        self.matcher = Matching([self.piece1, self.piece2, self.piece3])

    def test_find_matches_returns_list(self):
        matches = self.matcher.find_matches(threshold=0.5)
        self.assertIsInstance(matches, list)

    def test_matches_have_correct_keys(self):
        matches = self.matcher.find_matches(threshold=0.5)
        if matches:
            match = matches[0]
            expected_keys = {"piece_a", "edge_a", "piece_b", "edge_b", "score"}
            self.assertTrue(expected_keys.issubset(match.keys()))

    def test_matching_score_below_threshold(self):
        matches = self.matcher.find_matches(threshold=0.5)
        for match in matches:
            self.assertLess(match["score"], 0.5)

    def test_sorting_by_score(self):
        matches = self.matcher.find_matches(threshold=0.5)
        scores = [m["score"] for m in matches]
        self.assertEqual(scores, sorted(scores))

    def test_no_self_matches(self):
        matches = self.matcher.find_matches(threshold=0.5)
        for match in matches:
            self.assertNotEqual(match["piece_a"], match["piece_b"])

if __name__ == "__main__":
    unittest.main()
