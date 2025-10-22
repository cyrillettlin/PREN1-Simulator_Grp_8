from __future__ import print_function
from puzzle import puzzle

import cv2 as cv
import numpy as np

#Definiere Threshold
low_threshold = 100
high_threshold = 300
pathtofile = r""


#Bild
src = cv.imread(pathtofile)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Kanten finden
img_blur = cv.blur(src_gray, (3,3))
edges = cv.Canny(img_blur, low_threshold, high_threshold, 3)

# Konturen finden
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#Vier grösste Konturen: Alternativ
contours = sorted(contours, key=cv.contourArea, reverse=True)
largest_contours = contours[:4]

puzzle_pieces = [puzzle(cnt, i + 1) for i, cnt in enumerate(largest_contours)]

for piece in puzzle_pieces:
    print(piece)


#Output
output = src.copy()
colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255)]

for i, cnt in enumerate(largest_contours):
    color = colors[i % len(colors)]
    cv.drawContours(output, [cnt], -1, color, 3)
    M = cv.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv.contourArea(cnt)
        cv.putText(output, f"#{i+1} ({int(area)}px)", (cx - 40, cy + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
cv.putText(output, f"Low: {low_threshold}", (10, 30),
           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv.putText(output, f"High: {high_threshold}", (10, 60),
           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv.imshow("Vier groesste Konturen", output)
cv.waitKey(0)
cv.destroyAllWindows()
