import os
import math
import logging
import numpy as np
from edgedetection import EdgeDetection
from matching import Matching
from puzzleorganizer import PuzzleOrganizer
from visualizer import Visualizer
from GlobalArea import GlobalArea
from Position_and_Rotation.Rotation import Rotation
from Position_and_Rotation.Translation import Translation
from FlatEdgeFinder import FlatEdgeFinder
from Anchor import Anchor
from MatchPlacer import MatchPlacer
from puzzle import Puzzle
from cnnCornerDetector import cnnCornerDetector

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ── CNN einrichten (optional – Fallback auf BB wenn Modell fehlt) ──────────
cnn = cnnCornerDetector("modell/puzzle_model.onnx")
if cnn.available:
    Puzzle.set_cnn_detector(cnn)

# Puzzleteile einlesen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade_black_2.jpeg")
detector = EdgeDetection(path)
detector.load()

# ── Quellbild an Puzzle-Klasse weitergeben (für CNN-Ausschnitt) ────────────
Puzzle.set_source_image(detector.src)

# Konturen finden und filtern
detector.find_contours()
detector.filter_contours()

# Puzzleteile vorbereiten
pieces = detector.get_puzzle_pieces()
pieces_by_id = {p.index: p for p in pieces}

# ── Grid-Grösse automatisch bestimmen ─────────────────────────────────────
n_pieces  = len(pieces)
grid_size = math.ceil(math.sqrt(n_pieces))  # 4→2, 6→3, 9→3
logging.info(f"Teile gefunden: {n_pieces} → grid_size={grid_size}")

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

# ── Grid-Grösse automatisch übergeben ─────────────────────────────────────
organizer = PuzzleOrganizer(pieces, matches, grid_size=grid_size)
grid = organizer.organize()
logging.info("Puzzle-Layout:")
for row in grid:
    logging.info(f"  {row}")

# Instatntiate
ta  = Translation()
rot = Rotation()
ga  = GlobalArea()
ga.set_unsolved_puzzles(pieces)
ga.set_solved_puzzles(pieces)

# --- new helpers ---
flat_finder  = FlatEdgeFinder(num_points=100, logger=logging.getLogger("FlatEdgeFinder"))
#anchor       = Anchor(flat_finder=flat_finder, rot=rot, ta=ta, logger=logging.getLogger("Anchor"))
match_placer = MatchPlacer(ga=ga, rot=rot, ta=ta, logger=logging.getLogger("MatchPlacer"))

# optional debug
flat_finder.log_edge_types(pieces)

# choose anchor + get its flat edges
#anchor_piece, flat_edges = anchor.choose_anchor(ga.solved_puzzles)

# Scale and move unsolved pieces to match real world. This Needs to be done BEItFORE the Pieces are scaled.
# Otherwise the logic of identifying Puzzle edges would not work anymore.
ga.scale_all_puzzles(0.23, 0.23)
ga.translate_unsolved_puzzles(80, 190)

# Example: top-left directions & point computed from GlobalArea
target_corner_point = [ga.area_solved.x, ga.area_solved.y + ga.area_solved.h]

# choose corner directions/target point
#ang, dxdy = anchor.place_anchor(anchor_piece, flat_edges, (1, 0), (0, 1), target_corner_point)

# apply matches
matches_resorted = sorted(matches, key=lambda m: float(m["piece_a"]))
match_placer.apply_matches(matches_resorted)

# Visualisierung im Raster
Visualizer.show_all_edges_grid(pieces, image=detector.src)

# Visualisierung aller gefundenen Matches
#Visualizer.show_matches(matches, pieces)

# Show Solved Puzzle
#ga.show()