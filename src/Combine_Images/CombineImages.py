import cv2
import numpy as np
import os

def blend_multiple_images(images, alpha=None):
    if alpha is None:
        alpha = 1.0 / len(images)

    base_h, base_w = images[0].shape[:2]
    resized = [cv2.resize(img, (base_w, base_h)) for img in images]

    blended = np.zeros_like(resized[0], dtype=np.float32)
    for img in resized:
        blended += img.astype(np.float32) * alpha

    return np.clip(blended, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "Images")

    order = [6, 5, 4, 3, 2, 1]
    paths = [os.path.join(images_dir, f"Stiching_{i}.jpg") for i in order]

    images = [cv2.imread(p) for p in paths]

    for p, img in zip(paths, images):
        if img is None:
            print("Fehler beim Laden:", p)

    result = blend_multiple_images(images)

    output_path = os.path.join(images_dir, "blended_output.jpg")
    cv2.imwrite(output_path, result)
    print("Gespeichert als:", output_path)

    cv2.imshow("Blended 6 Images", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
