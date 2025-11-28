import cv2 as cv
import numpy as np
from edgedetection import EdgeDetection
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(BASE_DIR, "../Data/puzzle_real_example_1.jpg")
detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()

img = cv.imread(path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray, 4, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0,
 0, 255]


cv.namedWindow('dst', cv.WINDOW_NORMAL)
cv.resizeWindow('dst', 800, 600)
cv.imshow('dst', img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
