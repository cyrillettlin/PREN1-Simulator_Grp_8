from edgedetection import EdgeDetection
from puzzle import Puzzle


path = "Data/best_example.jpg"


detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.separate_contours()
detector.show_result()
