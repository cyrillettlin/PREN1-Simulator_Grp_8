import cv2
import os

def stitch_multiple_images(images):
    stitcher = cv2.Stitcher.create()
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print("Stitching failed, status:", status)
        return None


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "Images")

    order = [6, 5, 4, 3, 2, 1]
    paths = [os.path.join(images_dir, f"Stiching_{i}.jpg") for i in order]

    images = [cv2.imread(p) for p in paths]

    for p, img in zip(paths, images):
        if img is None:
            print("Fehler beim Laden:", p)

    stitched = stitch_multiple_images(images)

    if stitched is not None:
        output_path = os.path.join(images_dir, "panorama_output.jpg")
        cv2.imwrite(output_path, stitched)
        print("Gespeichert als:", output_path)

        cv2.imshow("Panorama 6-5-4-3-2-1", stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
