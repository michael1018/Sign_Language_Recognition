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

def test():
    # 读取图片
    src = cv2.imread('./dataset/test/m2_0.jpg', cv2.IMREAD_UNCHANGED)

    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)

    # 图像腐蚀处理
    erosion = cv2.erode(src, kernel)
    erosion = cv2.erode(erosion, kernel)

    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", erosion)
    cv2.imwrite('./dataset/test/erosion_m2_0.jpg', erosion)

    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = f'./dataset/test/erosion_and_dilation_test2.jpg'
    roi = cv2.imread(image_path)
    res = erosion_and_dilation(roi)
    cv2.imwrite('dataset/test/erosion_and_dilation_test3.jpg', res)
    # test()

