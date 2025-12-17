import os
from edgedetection import EdgeDetection
from puzzle import Puzzle
from puzzlematcher import PuzzleMatcher
import print_result
import visualizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade_black.jpeg")

detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()

#print_result.print_result(detector)

#detector.show_all_edges()

pieces = detector.get_puzzle_pieces()
for piece in pieces:
    piece.get_puzzle_edges()

matcher = PuzzleMatcher(pieces)
matches = matcher.find_matches(threshold=0.04) # Threshold => maximaler Score

# Nach Score absteigend sortieren
matches = sorted(matches, key=lambda m: m["score"], reverse=True)

print(f"Gefundene Matches: {len(matches)}")
if not matches:
    print("Keine Matches gefunden.")

for m in matches:
    print(f"Teil {m['piece_a']} Kante {m['edge_a']} ↔ "
          f"Teil {m['piece_b']} Kante {m['edge_b']} | Score={m['score']:.4f}")

visualizer.show_matches(matches, pieces)
