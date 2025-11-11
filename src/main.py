from edgedetection import EdgeDetection
from Puzzle import Puzzle


# path = "Data/Real-Puzzle.jpg"
# path = "Data/best_example.jpg"
path = "Data/Real-Puzzle_2.jpg"

detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.separate_contours()
detector.show_result()


