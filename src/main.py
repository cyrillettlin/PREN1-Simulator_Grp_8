from edgedetection import EdgeDetection
from puzzle import Puzzle


path = r"C:\Users\cyril\OneDrive - Hochschule Luzern\Documents\HSLU\3. Semester\PREN1\Simulator\experiments\Data\5776130955109141471.jpg"


detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.separate_contours()
detector.show_result()
