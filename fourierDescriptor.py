import cv2
import numpy as np

MIN_DESCRIPTOR = 32  # surprisingly enough, 2 descriptors are already enough


## Compute Fourier descriptors
def fourierDesciptor(res):
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    contour = find_contours(Laplacian)  # Extract contour point coordinates
    contour_array = contour[0][:, 0, :]  # Note that only the coordinates of the contour point with the largest area are retained here.
    ret_np = np.ones(dst.shape, np.uint8)  # Create a black frame
    ret = cv2.drawContours(ret_np, contour[0], -1, (255, 255, 255), 1)  # Draw white outline
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:, 0]
    contours_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contours_complex)
    # fourier_result = np.fft.fftshift(fourier_result)
    descirptor_in_use = truncate_descriptor(fourier_result)
    # reconstruct(ret, descirptor_in_use)
    return ret, descirptor_in_use


def find_contours(Laplacian):
    # binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    h = cv2.findContours(Laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    contour = h[0]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)  # 对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour


# 截短傅里叶描述子
def truncate_descriptor(fourier_result):
    descriptors_in_use = np.fft.fftshift(fourier_result)

    # 取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors_in_use = descriptors_in_use[low:high]

    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use


##由傅里叶描述子重建轮廓图
def reconstruct(img, descirptor_in_use):
    # descirptor_in_use = truncate_descriptor(fourier_result, degree)
    # descirptor_in_use = np.fft.ifftshift(fourier_result)
    # descirptor_in_use = truncate_descriptor(fourier_result)
    # print(descirptor_in_use)
    contour_reconstruct = np.fft.ifft(descirptor_in_use)
    contour_reconstruct = np.array([contour_reconstruct.real,
                                    contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)

    black_np = np.ones(img.shape, np.uint8)  # 创建黑色幕布
    black = cv2.drawContours(black_np, contour_reconstruct, -1, (255, 255, 255), -1)  # 绘制白色轮廓
    cv2.imshow("contour_reconstruct", black)
    cv2.imwrite('./dataset/test/recover.png',black)
    return black


if __name__ == '__main__':
    image_path = f'./dataset/test/1.jpg'
    roi = cv2.imread(image_path)
    ret, descirptor_in_use = fourierDesciptor(roi)
    cv2.imwrite('./dataset/test/fourierDesciptor.png', ret)
    print(descirptor_in_use)
    reconstruct(ret, descirptor_in_use)
    cv2.waitKey(0)