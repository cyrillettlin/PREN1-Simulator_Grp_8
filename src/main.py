from edgedetection import EdgeDetection
import os
import print_result
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# real example (black)
path = os.path.join(BASE_DIR, "../Data/puzzle_real_example_1.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_real_example_2.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_real_example_3.jpeg")

# 3D puzzle, self-made (colored)
# path = os.path.join(BASE_DIR, "../Data/puzzle_3d_1.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_3d_2.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_3d_3_flash.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_3d_unusable.jpg")


# self-made, paper (white)
# path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade_black.jpeg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade_black_2.jpeg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade_black_3.jpeg")


detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()
print_result.print_result(detector)

