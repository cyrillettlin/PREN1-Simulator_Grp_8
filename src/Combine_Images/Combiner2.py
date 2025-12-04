import cv2
import numpy as np
import os

#Aktuell beste Option muss jedoch punktgenau angrenzend sein.

def grid_2x3(images, target_width=400):
    top = images[0:3]
    bottom = images[3:6]

    def resize_row(row):
        resized = []
        for img in row:
            h = int(img.shape[0] * target_width / img.shape[1])
            resized.append(cv2.resize(img, (target_width, h)))
        return np.hstack(resized)

    top_row = resize_row(top)
    bottom_row = resize_row(bottom)

    max_width = max(top_row.shape[1], bottom_row.shape[1])

    def pad_to_width(img):
        diff = max_width - img.shape[1]
        if diff > 0:
            return np.hstack([img, np.zeros((img.shape[0], diff, 3), dtype=img.dtype)])
        return img

    top_row = pad_to_width(top_row)
    bottom_row = pad_to_width(bottom_row)

    return np.vstack([top_row, bottom_row])


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "Images")

    order = [6, 5, 4, 3, 2, 1]
    paths = [os.path.join(images_dir, f"Stiching_{i}.jpg") for i in order]

    images = [cv2.imread(p) for p in paths]

    for p, img in zip(paths, images):
        if img is None:
            print("Fehler beim Laden:", p)

    grid = grid_2x3(images)

    output_path = os.path.join(images_dir, "grid_output.jpg")
    cv2.imwrite(output_path, grid)
    print("Gespeichert als:", output_path)

    cv2.imshow("Grid 6 5 4 / 3 2 1", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
