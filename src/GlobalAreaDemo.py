from edgedetection import EdgeDetection
import matplotlib.pyplot as plt
import numpy as np
from GlobalArea import GlobalArea

#Set Image Path
# path = "Data/Real-Puzzle.jpg"
#path = "Data/Real-Puzzle_2.jpg"
path = "Data/best_example.jpg"

# Make Puzzles
detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()

#Optional, Show created contours.
detector.show_result()

# Get Puzzle Piceces.
puzzles = detector.get_puzzle_pieces()
#Instatntiate Global Area
ga = GlobalArea()
#Import the contours of the puzzle pieces.
ga.set_contour(puzzles)
# Scale the contours to the area of the 
ga.scale_contours(0.20,0.20)
ga.translate_contours(75,200)
ga.show()


