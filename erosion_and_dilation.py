import cv2
import numpy as np


def erosion_and_dilation(roi, show=False):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(roi, kernel)
    dilation = cv2.dilate(erosion, kernel)
    if show:
        cv2.imshow("erosion", erosion)
        cv2.imshow("dilation", dilation)
    return dilation


if __name__ == "__main__":
    image_path = f'./dataset/test/m4_0.jpg'
    roi = cv2.imread(image_path)
    erosion_and_dilation(roi)
    cv2.waitKey(0)
