import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.matching import matching


class TestContourComparison(unittest.TestCase):

    def test_matching_edges(self):
        a = PuzzleEdge(normalized_shape=np.array([0.1, 0.2, 0.3]))
        b = PuzzleEdge(normalized_shape=np.array([-0.3, -0.2, -0.1]))

        self.assertTrue(compare_contour(a, b))
        
#TODO: weitere Tests hinzufügen.
# Pip install im Readme dokumentieren.
