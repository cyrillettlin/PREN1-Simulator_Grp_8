from __future__ import print_function
from src.puzzle import puzzle

import cv2 as cv
import numpy as np
import argparse


#Vergleichsbild
srcexample = cv.imread("Data/6017330782235905198.jpg")
srcexample_gray = cv.cvtColor(srcexample, cv.COLOR_BGR2GRAY)

# Dein bestehendes Bild
src = cv.imread("Data/5990104323121597224.jpg")
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Kanten finden
edgesexample = cv.Canny(srcexample_gray, 50, 150)
edges = cv.Canny(src_gray, 50, 150)


# Konturen finden
contoursexample, hierarchy = cv.findContours(edgesexample, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#Vergleichkontour
sorted_contours = sorted(contoursexample, key=cv.contourArea, reverse=True)
examplecontour = sorted_contours[2] 

# Alle Zielkonturen nach Ähnlichkeit sortieren
similarities = []
for cnt in contours:
    score = cv.matchShapes(examplecontour, cnt, cv.CONTOURS_MATCH_I1, 0.0)
    similarities.append((score, cnt))

# Sortieren nach Score (kleiner = ähnlicher)
similarities.sort(key=lambda x: x[0])

best_matches = similarities[:4]


#Vier grösste Konturen: Alternativ
contours = sorted(contours, key=cv.contourArea, reverse=True)
largest_contours = contours[:4]

puzzle_pieces = [puzzle(cnt, i + 1) for i, (score, cnt) in enumerate(best_matches)]

for piece in puzzle_pieces:
    print(piece)



# Vergleichsbild mit Vergleichskontur markieren
output_example = srcexample.copy()
cv.drawContours(output_example, [examplecontour], -1, (0, 255, 0), 3)
cv.putText(output_example, "Vergleichskontur", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Zielbild mit den 4 ähnlichsten Konturen markieren
output_target = src.copy()
colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255)]

for i, (score, cnt) in enumerate(best_matches):
    color = colors[i % len(colors)]
    cv.drawContours(output_target, [cnt], -1, color, 3)
    M = cv.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv.putText(output_target, f"{i+1}", (cx - 10, cy + 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Beide Bilder anzeigen
cv.imshow("Vergleichsbild mit Vergleichskontur", output_example)
cv.imshow("Zielbild mit vier ähnlichsten Konturen", output_target)
cv.waitKey(0)
cv.destroyAllWindows()