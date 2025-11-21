from edgedetection import EdgeDetection
import os
import print_result

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# real example (black)
# path = os.path.join(BASE_DIR, "../Data/puzzle_real_example_1.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_real_example_2.jpg")

# 3D puzzle, self-made (colored)
# path = os.path.join(BASE_DIR, "../Data/puzzle_3d_1.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_3d_2.jpg")
path = os.path.join(BASE_DIR, "../Data/puzzle_3d_3_flash.jpg")
# path = os.path.join(BASE_DIR, "../Data/puzzle_3d_unusable.jpg")


# self-made, paper (white)
# path = os.path.join(BASE_DIR, "../Data/puzzle_selfmade.jpg")


detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()
print_result.print_result(detector)

