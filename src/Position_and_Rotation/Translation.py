import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2 as cv

# Waiting for GLobalArea to measure distances.
class Translation:

    def __init__(self):
        pass
        
    # Translate a contour.
    def translate_contour(contour, x_axis, y_axis):
        """
        Move a contour by x and y coordinates.
        """
        return contour + np.array([x_axis, y_axis])