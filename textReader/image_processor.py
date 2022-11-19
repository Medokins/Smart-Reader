import numpy as np
import cv2
from typing import Tuple 

BATCH = ['imgs']

class ImageProcessor:
    def __init__(self, image_arr: np.ndarray, image_size: Tuple[int, int], line_mode: bool = False) -> None:
        """
        :param image_arr: np.ndarray of image
        :param image_size: size to which the image will be scaled to
        :param line_mode: wheter or not there are many characters in an image (True) or just one (False)
        :return: None
        """
        self.image = image_arr
        self.image_size = image_size
        self.line_mode = line_mode

    def truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        From: https://github.com/githubharald/SimpleHTR/blob/master/src/preprocessor.py
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def process_img(self) -> None:
        """
        Scales image to self.image_size, convert image values to be in range [-1, 1] and transposes it
        :return None
        """
        # there are damaged files in IAM dataset - use black image instead
        if self.image is None:
            self.image = np.zeros(self.img_size[::-1])

        # image scaling
        self.image = self.image.astype(np.float)
        
        wt, ht = self.image_size
        h, w = self.image.shape
        f = np.minimum(wt / w, ht / h)
        tx = (wt - w * f) / 2
        ty = (ht - h * f) / 2

        # scale image into target image
        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones([ht, wt]) * 255
        self.image = cv2.warpAffine(self.image, M, dsize=(wt, ht), dst = target, borderMode = cv2.BORDER_TRANSPARENT)

        # transpose for Tensor Flow
        self.image = cv2.transpose(self.image)

        # convert to range [-1, 1]
        self.image /= np.max(np.abs(self.image), axis=0)


def main():
    import matplotlib.pyplot as plt

    img = cv2.imread('data/img/a01/a01-000u/a01-000u-00-05.png', cv2.IMREAD_GRAYSCALE)
    img_scaled = ImageProcessor(img, (256, 64))
    img_scaled.process_img()

    _, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img, cmap='gray')
    axarr[1].imshow(np.transpose(img_scaled.image), cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()