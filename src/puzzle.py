import cv2 as cv
import numpy as np
import argparse

class puzzle:

    def __init__(self, contour, index):
        self.index = index
        self.contour = contour
        self.area = cv.contourArea(contour)
        self.bounding_box = cv.boundingRect(contour)
        self.centre_point
        #CenterPoint

    def get_puzzle_edges():
        # Gedrehte Bounding Box um das Puzzleteil
        rect = cv.minAreaRect(self.contour)
        box = cv.boxPoints(rect)
        box = np.intp(box)

    # Kanten in Reihenfolge: oben, rechts, unten, links
        edges = []
        for i in range(4):
            p1 = tuple(box[i])
            p2 = tuple(box[(i + 1) % 4])
            edges.append((p1, p2))
        return edges, box
    
        #Kanten ansprechen im Vergleichsverfahren --> Laura
        #edges, box = get_puzzle_edges(contour)
        #top_edge = edges[0]     
        #right_edge = edges[1]
        #bottom_edge = edges[2]
        #left_edge = edges[3]

    def get_center_point():
        #calculation
        M = cv.moments(self.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
            #Auf Polarform anpassen
        else:
            return (0, 0)
    

    def __repr__(self):
        x, y, w, h = self.bounding_box
        return f"PuzzlePiece {self.index}: Fläche={self.area:.2f}, Box=({x},{y},{w},{h})"
    

   
