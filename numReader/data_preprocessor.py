import cv2
import numpy as np
from PIL import Image

# helper function to check images during coding
def showProcessedImage(name):
    converted = preprocessImage(name)
    cv2.imshow('binarized', cv2.resize(converted*255, (256, 256), interpolation=None))
    cv2.waitKey(0)

def preprocessImage(name) -> np.ndarray:
    im = cv2.imread(f'handwrittenNumbers/{name}.png', cv2.IMREAD_GRAYSCALE)
    im[im != 255] = 1.0
    im[im == 255] = 0.0

    tf_image = np.empty((1, 28, 28), dtype=np.double)
    tf_image[0] = im
    return tf_image

def getBoundingBoxes(name) -> np.ndarray:
    """
    :param string name: name of file (with 2 or more digits numbers) in handwrittenNumbers directory
    :return np.ndarray: individual bounding boxes areas
    """
    pass

if __name__ == '__main__':
    preprocessImage(3)