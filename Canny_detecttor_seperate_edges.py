from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

# Dein bestehendes Bild
src = cv.imread("Data/5990104323121597224.jpg")
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Kanten finden
edges = cv.Canny(src_gray, 50, 150)

# Konturen finden
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Nur die 4 grössten Contours behalten.
contours = sorted(contours, key=cv.contourArea, reverse=True)[:4]

# Kopie für die Visualisierung
output = src.copy()

# Alle Konturen zeichnen
cv.drawContours(output, contours, -1, (0,255,0), 2)

print(f"Anzahl erkannter Objekte: {len(contours)}")

# Optional: einzelne Objekte ansprechen
for i, contour in enumerate(contours):
    # Bounding Box
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv.putText(output, f"Objekt {i+1}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

for i, c in enumerate(contours):
    area = cv.contourArea(c)
    perimeter = cv.arcLength(c, True)
    #print(f"Objekt {i+1}: Fläche={area:.2f}, Umfang={perimeter:.2f}")


for i in range(len(contours)):
    for j in range(i+1, len(contours)):
        similarity = cv.matchShapes(contours[i], contours[j], cv.CONTOURS_MATCH_I1, 0.0)
        #print(f"Ähnlichkeit Objekt {i+1} ↔ Objekt {j+1}: {similarity:.4f}")


#Vergleich nach Ähnlichkeit und/oder Flächengrösse

cv.imshow("Objekte", output)
cv.waitKey(0)
cv.destroyAllWindows()