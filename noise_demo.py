import cv2
import numpy as np
import os


for folder in os.listdir("data/trainingSet"):
    images = os.listdir(f"data/trainingSet/{folder}")[:1]
    for i, image_name in enumerate(images):
        image_path = f"data/trainingSet/{folder}/{image_name}"
        pixels = cv2.imread(image_path, 0).astype("float64") / 255.
        cv2.imshow("Image without noise", cv2.resize(pixels, (256, 256)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pixels += np.random.uniform(-0.3, 0.3, pixels.shape)
        pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))
        cv2.imshow("Image with noise", cv2.resize(pixels, (256, 256)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
