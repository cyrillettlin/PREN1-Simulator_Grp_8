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
dst = cv.cornerHarris(gray, 8, 3, 0.04) #4,3,0.04

# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image, findet lokales maxima
img[dst > 0.01 * dst.max()] = [0,
 0, 255]




scaled_output = cv.resize(img, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
cv.imshow('dst', scaled_output)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
