from edgedetection import EdgeDetection
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path = os.path.join(BASE_DIR, "../Data/Real-Puzzle.jpg"
# path = os.path.join(BASE_DIR, "../Data/best_example.jpg"
# path = os.path.join(BASE_DIR, "../Data/Real-Puzzle_2.jpg"
path = os.path.join(BASE_DIR, "../Data/Real-Puzzle_2.jpg")

detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.separate_contours()
detector.show_result()
