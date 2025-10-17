import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a new figure and axis
fig, ax = plt.subplots()

# Set equal aspect ratio so shapes aren’t distorted
ax.set_aspect('equal')

# Example shapes:
# Rectangle: (x, y) is bottom-left corner
rect = patches.Rectangle((1, 1), 2, 1, linewidth=2, edgecolor='blue', facecolor='lightblue')

# Circle: (x, y) is center
circle = patches.Circle((5, 2), radius=1, linewidth=2, edgecolor='red', facecolor='pink')

# Ellipse: (x, y) is center
ellipse = patches.Ellipse((3, 4), width=3, height=1.5, linewidth=2, edgecolor='green', facecolor='lightgreen')

# Polygon: list of (x, y) points
polygon = patches.Polygon([[6,4], [7,5], [8,4], [7,3]], closed=True, linewidth=2, edgecolor='purple', facecolor='lavender')

# Add shapes to the plot
ax.add_patch(rect)
ax.add_patch(circle)
ax.add_patch(ellipse)
ax.add_patch(polygon)

# Set limits of the canvas
ax.set_xlim(0, 14.14)
ax.set_ylim(0, 10)

# Show the drawing
plt.show()
