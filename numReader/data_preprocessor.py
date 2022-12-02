import cv2
import numpy as np
import tensorflow as tf
from settings import *

def preprocessImage(imageArray: np.ndarray) -> np.ndarray:
    imageArray = cv2.resize(imageArray, (28, 28), interpolation = cv2.INTER_NEAREST)
    imageArray[imageArray != 255] = 1.0
    imageArray[imageArray == 255] = 0.0

    tf_image = np.empty((1, 28, 28), dtype=np.double)
    tf_image[0] = imageArray
    return tf_image

def getBoundingBoxes(name: str, visualize: bool = False) -> np.ndarray:
    """
    :param string name: name of file (with 2 or more digits numbers) in handwrittenNumbers directory
    :return np.ndarray: individual bounding boxes areas
    """
    im = cv2.imread(f'handwrittenNumbers/{name}.png', cv2.IMREAD_GRAYSCALE)
    im = ~im
    binaryIm = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -1)

    # get individual objects, fill in some noise
    num_of_objects, labeledImage, componentStats, _ = cv2.connectedComponentsWithStats(binaryIm)
    remainingComponentLabels = [i for i in range(1, num_of_objects) if componentStats[i][4] >= MIN_AREA_FILTER]
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    # final clearing of any artefacts by closing
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
    closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, CLOSING_INTERATIONS, cv2.BORDER_REFLECT101)

    # bounding boxes
    contours, hierarchy = cv2.findContours(closingImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = []

    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))

    # Draw the bounding boxes on the binarized input image:
    if visualize:
        imCopy = cv2.imread(f'handwrittenNumbers/{name}.png')
        for i in range(len(boundRect)):
            color = (0, 255, 0)
            cv2.rectangle(imCopy, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        cv2.imshow(imCopy)
        cv2.waitKey(0)

    digits_array = []

    for i in range(len(boundRect)):
        # Get the start coordinates and size of bounding box
        x_start, y_start, width, height = boundRect[i]
        x_end = x_start + width
        y_end = y_start + height

        croppedImg = closingImage[y_start:y_end, x_start:x_end]
        digits_array.append(croppedImg)
    
    return digits_array


if __name__ == '__main__':
    boundingBoxes = getBoundingBoxes('boundingBoxTest')
    im = preprocessImage(boundingBoxes[1])[0]
    # this doesnt work properly, will fix it tommorow (it creates blank image, probably due to binarization in bounding boxes)
    convertedImage = preprocessImage(im)
    cv2.imshow('test', convertedImage[0])
    cv2.waitKey(0)
    new_model = tf.keras.models.load_model('numReader.model')
    predictions = new_model.predict([convertedImage])
    print(np.argmax(predictions[0]))