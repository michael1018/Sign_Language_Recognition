import cv2
import numpy as np

image_path = f'./dataset/test/0.jpg'


def method1_RGB(roi):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # 转换到RGB空间
    (R, G, B) = cv2.split(rgb)  # 获取图像每个像素点的RGB的值，即将一个二维矩阵拆成三个二维矩阵
    skin = np.zeros(R.shape, dtype=np.uint8)  # 掩膜
    (x, y) = R.shape  # 获取图像的像素点的坐标范围
    for i in range(0, x):
        for j in range(0, y):
            # 判断条件，不在肤色范围内则将掩膜设为黑色，即255
            if R[i][j] > 95 and G[i][j] > 40 and B[i][j] > 20:
                skin[i][j] = 255

    res = cv2.bitwise_and(roi, roi, mask=skin)  # 图像与运算
    cv2.imshow("res", res)
    return res


def method2_HSV(roi):
    low = np.array([0, 48, 50])  # 最低阈值
    high = np.array([20, 255, 255])  # 最高阈值
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转换到HSV空间
    mask = cv2.inRange(hsv, low, high)  # 掩膜，不在范围内的设为255
    res = cv2.bitwise_and(roi, roi, mask=mask)  # 图像与运算
    return res


def method3_YCrCb(roi):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)  # 绘制椭圆弧线
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, Cr, Cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    skin = np.zeros(Cr.shape, dtype=np.uint8)  # 掩膜
    (x, y) = Cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if skinCrCbHist[Cr[i][j], Cb[i][j]] > 0:  # 若不在椭圆区间中
                skin[i][j] = 255
    res = cv2.bitwise_and(roi, roi, mask=skin)
    return res


def method4_Otsu(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = cv2.bitwise_and(roi, roi, mask=skin)
    return res


def method5_Cr_Cb(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            # 每个像素点进行判断
            if (cr[i][j] > 130) and (cr[i][j] < 175) and (cb[i][j] > 77) and (cb[i][j] < 127):
                skin[i][j] = 255
    res = cv2.bitwise_and(roi, roi, mask=skin)
    return res


if __name__ == '__main__':
    roi = cv2.imread(image_path)
    res_1 = method1_RGB(roi)
    res_2 = method2_HSV(roi)
    res_3 = method3_YCrCb(roi)
    res_4 = method4_Otsu(roi)
    res_5 = method5_Cr_Cb(roi)
    cv2.imwrite('./dataset/test/m1_0.jpg', res_1)
    cv2.imwrite('./dataset/test/m2_0.jpg', res_2)
    cv2.imwrite('./dataset/test/m3_0.jpg', res_3)
    cv2.imwrite('./dataset/test/m4_0.jpg', res_4)
    cv2.imwrite('./dataset/test/m5_0.jpg', res_5)

