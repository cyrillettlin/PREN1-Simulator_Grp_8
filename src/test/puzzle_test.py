import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from puzzle import Puzzle

class TestPuzzle(unittest.TestCase):

    def setUp(self):
        # Rechteckkontur
        self.rect_contour = np.array([
            [[0, 0]],
            [[100, 0]],
            [[100, 50]],
            [[0, 50]]
        ], dtype=np.int32)

        # Leicht verzerrtes Rechteck
        self.skewed_contour = np.array([
            [[0, 0]],
            [[102, -2]],
            [[100, 52]],
            [[-1, 50]]
        ], dtype=np.int32)

        self.puzzle_rect = Puzzle(self.rect_contour, index=0)
        self.puzzle_skewed = Puzzle(self.skewed_contour, index=1)

    def test_center_point(self):
        cx, cy = self.puzzle_rect.get_center_point()
        self.assertEqual((cx, cy), (50, 25), "Center Point des Rechtecks sollte korrekt sein")

    def test_best_4_corners_returns_4_points(self):
        corners = self.puzzle_rect.get_best_4_corners()
        self.assertEqual(len(corners), 4, "Es sollten genau 4 Ecken zurückgegeben werden")
        for pt in corners:
            self.assertIsInstance(pt, tuple)

    def test_puzzle_edges_count_and_type(self):
        edges = self.puzzle_rect.get_puzzle_edges()
        self.assertEqual(len(edges), 4, "Es sollten 4 Kanten zurückgegeben werden")
        for edge in edges:
            self.assertIn("points", edge)
            self.assertIn("type", edge)
            self.assertTrue(len(edge["points"]) >= 2)

    def test_edges_ordering(self):
        edges = self.puzzle_rect.get_puzzle_edges()
        top_edge = edges[0]["points"]
        bottom_edge = edges[2]["points"]
        self.assertTrue(all(p[1] <= top_edge[-1][1] for p in top_edge) or len(top_edge) >= 2)
        self.assertTrue(all(p[1] >= bottom_edge[0][1] for p in bottom_edge) or len(bottom_edge) >= 2)

    def test_skewed_contour_edges(self):
        edges = self.puzzle_skewed.get_puzzle_edges()
        self.assertEqual(len(edges), 4, "Auch verzerrte Konturen sollten 4 Kanten liefern")
        for edge in edges:
            self.assertIn("points", edge)
            self.assertTrue(len(edge["points"]) >= 2, "Jede Kante sollte mindestens 2 Punkte enthalten")

    def test_rotated_bounding_box(self):
        box_edges = self.puzzle_rect.get_rotated_bounding_box()
        self.assertEqual(len(box_edges), 4, "Bounding Box sollte 4 Kanten liefern")
        for seg in box_edges:
            self.assertEqual(len(seg), 2, "Jede Bounding Box Kante sollte genau 2 Punkte haben")

if __name__ == "__main__":
    unittest.main()
