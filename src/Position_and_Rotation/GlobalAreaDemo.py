from ..edgedetection import EdgeDetection
import matplotlib.pyplot as plt
import numpy as np
from src.GlobalArea import GlobalArea

#Set Image Path
# path = "Data/puzzle_real_example_1.jpg"
#path = "Data/puzzle_real_example_2.jpg"
path = "Data/puzzle_selfmade.jpg"

# Make Puzzles
detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()

#Optional, Show created contours.
# detector.show_result()

# Get Puzzle Piceces.
puzzles = detector.get_puzzle_pieces()
#Instatntiate Global Area
ga = GlobalArea()
#Import the contours of the puzzle pieces.
ga.set_unsolved_contours(puzzles)
ga.set_solved_contours(puzzles)
# Scale tcontourshe contours to the area of the 
ga.scale_all_contours(0.20,0.20)
ga.translate_unsolved_contours(75,200)
ga.show()


