import cv2
import numpy as np
import os
def blend_images(img1, img2, alpha=0.5):
    # Resize second image to match the first
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Blend (alpha * img1 + (1 - alpha) * img2)
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    return blended

def combine_images(image1, image2):
    # Resize both images to have the same smaller dimensions
    target_width = 400  # Set your desired width here
    target_height = int(image1.shape[0] * target_width / image1.shape[1])
    
    image1 = cv2.resize(image1, (target_width, target_height))
    image2 = cv2.resize(image2, (target_width, target_height))
    
    # Combine the images horizontally
    combined_image = np.hstack((image1, image2))
    
    return combined_image

if __name__ == "__main__":


    script_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(script_dir, "Images/Duo1.jpg")
    img2_path = os.path.join(script_dir, "Images/Duo2.jpg")

    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)


    # Combine the images horizontally
    combined_image = combine_images(image1, image2)

    # Show the combined image
    '''
    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(0) #0 to exit
    cv2.destroyAllWindows()
    '''

    combined = blend_images(image1, image2, alpha=0.7)
    cv2.imshow("Overlap", combined)
    cv2.waitKey(0) #0 to exit
    cv2.destroyAllWindows()
