import cv2
import numpy as np


def edge_detection(res, show=False):
    binaryimg = cv2.Canny(res, 50, 200)
    h = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # find edge
    contours = h[0]  # get edge
    ret = np.ones(res.shape, np.uint8)  # create graph
    cv2.drawContours(ret, contours, -1, (255, 255, 255), 1)  # draw white line
    if show:
        cv2.imshow("ret", ret)
    return ret


if __name__ == '__main__':
    image_path = f'./dataset/test/0.jpg'
    roi = cv2.imread(image_path)
    edge_detection(roi,show=True)
    cv2.waitKey(0)
