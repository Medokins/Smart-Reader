import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from typing import Tuple

class ImageProcessor:
    def __init__(self, image_arr: np.ndarray) -> None:
        """
        :param image_arr: grayscale np.ndarray of image
        :return: None
        """
        self.image = image_arr
        self.preprocess_image()

    def preprocess_image(self) -> None:
        """
        cuts given word (from an image) into seperate characters images
        """
        # Here I want to cut given word into seperate characters
        blurred = cv2.GaussianBlur(self.image, (5, 5), 0)
        _, self.image = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
        edged = cv2.Canny(blurred, 30, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print(cnts[0])
        #cnts = sort_contours(cnts, method="left-to-right")[0]

        cv2.imshow('test-img', self.image)
        cv2.waitKey(0)

def main():
    path = 'data/img/a02/a02-000/a02-000-00-05.png'
    ImageProcessor(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

if __name__ == '__main__':
    main()