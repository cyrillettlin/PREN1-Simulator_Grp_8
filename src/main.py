import os
import logging
import numpy as np
from edgedetection import EdgeDetection
from edgecomparator import EdgeComparator
from puzzle import Puzzle
from matching import Matching
from puzzleorganizer import PuzzleOrganizer
from visualizer import Visualizer
from GlobalArea import GlobalArea
from Position_and_Rotation.Rotation import Rotation
from Position_and_Rotation.Translation   import Translation

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

#Scale and move unsolved pieces to match real world.
ga.scale_all_puzzles(0.23, 0.23)
ga.translate_unsolved_puzzles(80,190)

#Solve the Puzzle!-----------------------------------------
#Sort matches to be moved in order.
matches_resorted = sorted(matches, key=lambda m: float(m["piece_a"]))  #First Piece First

#Find Flat edges of anchor piece to align with area_solved corner. 
def classify_edge_points(edge_points, num_points=100):
    comp = EdgeComparator(edge_points, edge_points, num_points=num_points)
    norm = comp._normalize_geometry(comp.edge_a)
    res  = comp._resample_edge(norm)
    return comp.get_edge_type(res)

def edge_types(piece):
    piece.get_puzzle_edges()
    return [(i, classify_edge_points(e["points"])) for i, e in enumerate(piece.edges)]

for p in pieces:
    logging.info(f"Piece {p.index} edge types: {edge_types(p)}")
types = edge_types(piece)  
# e.g. [(0,'flat'), (1,'tab'), (2,'hole'), (3,'flat')]

flat_edges = [i for i, t in types if t == "flat"]

#Rotate and move anchor_piece to align with top-left corner. 
anchor_piece = ga.get_solved_puzzle_piece(0)
ang = rot.anchor_rotation_for_corner_deg(anchor_piece, flat_edges,(-1,0), (0,1))
rot.rotate_puzzle_in_place(anchor_piece, ang)
trans = ta.delta_xy(ga.get_solved_puzzle_piece(0).corners[0],[120,168])
ta.translate_puzzle_in_place(ga.get_solved_puzzle_piece(0),trans)

#Rotate and move the rest of the pieces to align with the anchor_piece. 
if not matches:
    logging.warning("Keine Matches gefunden.")
else:
    for m in matches:
        req_rot = rot.compute_required_rotation_deg(ga.get_matching_edge_lines(m['piece_a'],m['edge_a'],m['piece_b'],m['edge_b']))
        logging.info(f"Required Rotation: {req_rot}")
        rot.rotate_puzzle_in_place(ga.get_solved_puzzle_piece(m['piece_b']-1),req_rot)
        ta.translate_piece_b_to_a_in_place(ga.get_solved_puzzle_piece(m['piece_b']-1),ga.get_matching_edge_lines(m['piece_a'],m['edge_a'],m['piece_b'],m['edge_b']))


# Visualisierung im Raster  
#Visualizer.show_all_edges_grid(pieces, image=detector.src)

# Visualisierung aller gefundenen Matches
#Visualizer.show_matches(matches, pieces)
#Show Solved Puzzle
ga.show()