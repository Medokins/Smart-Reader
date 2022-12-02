import cv2
import numpy as np
from settings import *

def preprocessImage(imageArray: np.ndarray) -> np.ndarray:
    # filing border with white pixels untill it's a square to not distort image while resizing later
    length = max(imageArray.shape[0:2])
    squared_image = np.empty((length, length), np.uint8)
    squared_image.fill(255)
    ax,ay = (length - imageArray.shape[1]) // 2, (length - imageArray.shape[0])//2
    squared_image[ay:imageArray.shape[0] + ay, ax:ax + imageArray.shape[1]] = imageArray

    final_image = cv2.resize(imageArray, (28, 28))
    # I probably need to skeletonize/thinner image here
    final_image[final_image != 255] = 1.0
    final_image[final_image == 255] = 0.0

    tf_image = np.empty((1, 28, 28), dtype=np.double)
    tf_image[0] = final_image
    return tf_image

def getBoundingBoxes(name: str, visualize: bool = False) -> np.ndarray:
    """
    :param string name: name of file (with 2 or more digits numbers) in handwrittenNumbers directory
    :return np.ndarray: coordinates of bounding boxes -> [[x_start, y_start, x_end, y_end], [], ...]
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
        cv2.imshow('bounding boxes',imCopy)
        cv2.waitKey(0)

    coordinates_array = np.empty((len(boundRect), 4), dtype=np.int16)

    for i in range(len(boundRect)):
        # Get the start coordinates and size of bounding box
        x_start, y_start, width, height = boundRect[i]

        coordinates_array[i][0] = x_start - 3
        coordinates_array[i][1] = y_start - 2
        coordinates_array[i][2] = x_start + width + 3
        coordinates_array[i][3] = y_start + height + 3
    
    return coordinates_array


if __name__ == '__main__':
    name = '7'
    im = cv2.imread(f'handwrittenNumbers/{name}.png', cv2.IMREAD_GRAYSCALE)
    boundingBoxes = getBoundingBoxes(name, visualize=False)

    # sort bounding boxes in order from left to right
    sortedBoundigBoxes = boundingBoxes[boundingBoxes[:,0].argsort()]

    # single_digit = im[y_start:y_end, x_start:x_end]
    index = 0
    single_digit = im[sortedBoundigBoxes[index][1]:sortedBoundigBoxes[index][3],
                      sortedBoundigBoxes[index][0]:sortedBoundigBoxes[index][2]]

    boundingBoxConverted = preprocessImage(single_digit)
    
    # # for testing

    # import pandas as pd
    # pd.DataFrame(cv2.resize(single_digit, (28, 28))).to_csv('before.csv')
    # pd.DataFrame(boundingBoxConverted[0]).to_csv('after.csv')

    import tensorflow as tf

    new_model = tf.keras.models.load_model('numReader.model')

    im = cv2.imread(f'handwrittenNumbers/{name}.png', cv2.IMREAD_GRAYSCALE)
    convertedImage = preprocessImage(im)
    predictions = new_model.predict([convertedImage])
    print('Original image preprocessed', np.argmax(predictions[0]))

    predictions = new_model.predict([boundingBoxConverted])
    print('BoundingBox image preprocessed', np.argmax(predictions[0]))

    cv2.imshow('original', convertedImage[0])
    cv2.waitKey(0)
    cv2.imshow('after bounding box', boundingBoxConverted[0])
    cv2.waitKey(0)
