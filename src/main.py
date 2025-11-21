from edgedetection import EdgeDetection
import os
import print_result

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(BASE_DIR, "../Data/Real-Puzzle.jpg")
# path = os.path.join(BASE_DIR, "../Data/best_example.jpg")
# path = os.path.join(BASE_DIR, "../Data/Real-Puzzle_2.jpg")
# path = os.path.join(BASE_DIR, "../Data/Puzzle_1_print.jpg")

detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()
print_result.print_result(detector)

