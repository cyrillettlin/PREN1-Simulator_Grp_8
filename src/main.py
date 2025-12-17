import os
import logging
from edgedetection import EdgeDetection
from puzzle import Puzzle
from matching import Matching
from puzzleorganizer import PuzzleOrganizer
from visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Puzzleteile einlesen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade_black.jpeg")

detector = EdgeDetection(path)
detector.load()

# Konturen finden und filtern
detector.find_contours()
detector.filter_contours()

# Puzzleteile vorbereiten
pieces = detector.get_puzzle_pieces()
for piece in pieces:
    piece.get_puzzle_edges()
    logging.info(f"Teil {piece.index} Kanten: {[len(e['points']) for e in piece.edges]}")

# Matches finden
matcher = Matching(pieces)
matches = matcher.find_matches(threshold=0.04)
matches = sorted(matches, key=lambda m: m["score"], reverse=True)
logging.info(f"Gefundene Matches: {len(matches)}")

if not matches:
    logging.warning("Keine Matches gefunden.")
else:
    for m in matches:
        logging.info(f"Teil {m['piece_a']} Kante {m['edge_a']} ↔ "
                     f"Teil {m['piece_b']} Kante {m['edge_b']} | Score={m['score']:.4f}")

# Puzzle in 2x2 Array legen
organizer = PuzzleOrganizer(pieces, matches, grid_size=2)
grid = organizer.organize()
logging.info("Puzzle-Layout:")
for row in grid:
    logging.info(f"  {row}")

# Visualisierung im Raster
Visualizer.show_all_edges_grid(pieces, image=detector.src)
