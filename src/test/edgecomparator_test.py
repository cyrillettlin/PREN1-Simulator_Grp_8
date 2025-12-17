import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np


from edgecomparator import EdgeComparator


class TestEdgeComparator(unittest.TestCase):

    def setUp(self):
        # Einfache Beispielkanten
        self.tab_edge = np.array([[0, 0], [0.5, 0.2], [1, 0]])
        self.hole_edge = np.array([[0, 0], [0.5, -0.2], [1, 0]])
        self.flat_edge = np.array([[0, 0], [0.5, 0], [1, 0]])

    def test_tab_matches_hole(self):
        comparator = EdgeComparator(self.tab_edge, self.hole_edge)
        score = comparator.compare()
        self.assertLess(score, 1.0,)

    def test_tab_matches_tab(self):
        comparator = EdgeComparator(self.tab_edge, self.tab_edge)
        score = comparator.compare()
        self.assertEqual(score, 98.0,)

    def test_flat_edge_penalty(self):
        comparator = EdgeComparator(self.flat_edge, self.tab_edge)
        score = comparator.compare()
        self.assertEqual(score, 99.0,)

    def test_resampling_preserves_length(self):
        comparator = EdgeComparator(self.tab_edge, self.hole_edge, num_points=50)
        resampled = comparator._resample_edge(self.tab_edge)
        self.assertEqual(resampled.shape[0], 50, "Resampling sollte korrekte Punktanzahl liefern")

    def test_normalization_rotates_and_scales(self):
        comparator = EdgeComparator(self.tab_edge, self.hole_edge)
        normalized = comparator._normalize_geometry(self.tab_edge)
        self.assertAlmostEqual(normalized[-1, 0], 1.0, places=6, msg="Normierte Kante sollte x=1 am Ende haben")
        self.assertAlmostEqual(normalized[0, 0], 0.0, places=6, msg="Normierte Kante sollte x=0 am Anfang haben")

if __name__ == "__main__":
    unittest.main()
        
