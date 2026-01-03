import os
import logging
import numpy as np
from edgedetection import EdgeDetection
from matching import Matching
from puzzleorganizer import PuzzleOrganizer
from visualizer import Visualizer
from GlobalArea import GlobalArea
from Position_and_Rotation.Rotation import Rotation
from Position_and_Rotation.Translation   import Translation
from FlatEdgeFinder import FlatEdgeFinder
from Anchor import Anchor
from MatchPlacer import MatchPlacer

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
pieces_by_id = {p.index: p for p in pieces}
for piece in pieces:
    piece.get_puzzle_edges()
    logging.info(f"Teil {piece.index} Kanten: {[len(e['points']) for e in piece.edges]}")

# Matches finden
matcher = Matching(pieces)
matches = matcher.find_matches(threshold=0.04)
logging.info(f'matches: {matches}')
matches = sorted(matches, key=lambda m: float(m["score"]))  # best first
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





#Instatntiate 
ta = Translation()
rot = Rotation()
ga = GlobalArea()
ga.set_unsolved_puzzles(pieces) 
ga.set_solved_puzzles(pieces)



# --- new helpers ---
flat_finder = FlatEdgeFinder(num_points=100, logger=logging.getLogger("FlatEdgeFinder"))
anchor = Anchor(flat_finder=flat_finder, rot=rot, ta=ta, logger=logging.getLogger("Anchor"))
match_placer = MatchPlacer(ga=ga, rot=rot, ta=ta, logger=logging.getLogger("MatchPlacer"))

# optional debug
flat_finder.log_edge_types(pieces)

# choose anchor + get its flat edges

anchor_piece, flat_edges = anchor.choose_anchor(ga.solved_puzzles)
#Scale and move unsolved pieces to match real world. This Needs to be done BEItFORE the Pieces are scaled. 
#Otherwise the logic of identifying Puzzle edges would not work anymore. 
ga.scale_all_puzzles(0.23, 0.23)
ga.translate_unsolved_puzzles(80,190)


# Example: top-left directions & point computed from GlobalArea

target_corner_point = [ga.area_solved.x, ga.area_solved.y + ga.area_solved.h]

# choose corner directions/target point
ang, dxdy = anchor.place_anchor(anchor_piece, flat_edges, (1,0), (0,1), target_corner_point)


# apply matches
matches_resorted = sorted(matches, key=lambda m: float(m["piece_a"]))
match_placer.apply_matches(matches_resorted)




# Visualisierung im Raster  
Visualizer.show_all_edges_grid(pieces, image=detector.src)

# Visualisierung aller gefundenen Matches
Visualizer.show_matches(matches, pieces)
#Show Solved Puzzle
ga.show()