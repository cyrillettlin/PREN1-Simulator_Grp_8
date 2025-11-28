import cv2 as cv
import numpy as np
from edgedetection import EdgeDetection
import os
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(BASE_DIR, "../Data/puzzle_real_example_1.jpg")
detector = EdgeDetection(path)
detector.load()
detector.find_contours()
detector.filter_contours()

img = cv.imread(path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = corners.astype(int)

for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)

plt.imshow(img), plt.show()