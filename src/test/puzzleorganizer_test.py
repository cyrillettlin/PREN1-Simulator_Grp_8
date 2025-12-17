import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from collections import defaultdict
from edgedetection import EdgeDetection
from puzzle import Puzzle
from puzzlematcher import PuzzleMatcher
from puzzleorganizer import PuzzleOrganizer

class TestPuzzleOrganization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #Initialisierung: Kanten erkennen, Puzzleteile vorbereiten
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(BASE_DIR, "../../Data/puzzle_selfmade_black.jpeg")

        cls.detector = EdgeDetection(path)
        cls.detector.load()
        cls.detector.find_contours()
        cls.detector.filter_contours()

        cls.pieces = cls.detector.get_puzzle_pieces()
        for piece in cls.pieces:
            piece.get_puzzle_edges()

        cls.matcher = PuzzleMatcher(cls.pieces)
        cls.matches = cls.matcher.find_matches(threshold=0.04)
        cls.matches = sorted(cls.matches, key=lambda m: m["score"], reverse=True)

        cls.organizer = PuzzleOrganizer(cls.pieces, cls.matches, grid_size=2)
        cls.grid = cls.organizer.organize()

    def test_matches_found(self):
        #Testet, ob Matches gefunden wurden
        self.assertGreater(len(self.matches), 0, "Keine Matches gefunden")

    def test_layout_consistency(self):
        #Testet die Konsistenz der Matches im Layout
        for m in self.matches:
            a, b = m["piece_a"], m["piece_b"]
            if a not in self.organizer.positions or b not in self.organizer.positions:
                continue
            pos_a = self.organizer.positions[a]
            pos_b = self.organizer.positions[b]
            c_edge = self.organizer.EDGE_DIR[m["edge_a"]]
            dr, dc = self.organizer.OFFSETS[c_edge]
            expected_pos = (pos_a[0]+dr, pos_a[1]+dc)
            self.assertEqual(pos_b, expected_pos, f"Inkonsistenz gefunden: {m}")

    def test_layout_stability(self):
        #Prüft, ob das Layout über mehrere Läufe stabil bleibt
        layouts = []
        for _ in range(10):
            matcher_temp = PuzzleMatcher(self.pieces)
            matches_temp = matcher_temp.find_matches(threshold=0.04)
            matches_temp = sorted(matches_temp, key=lambda m: m["score"], reverse=True)
            organizer_temp = PuzzleOrganizer(self.pieces, matches_temp, grid_size=2)
            grid_temp = organizer_temp.organize()
            layouts.append(tuple(tuple(row) for row in grid_temp))
        self.assertEqual(len(set(layouts)), 1, "Layout variiert über mehrere Läufe")

if __name__ == "__main__":
    unittest.main()
