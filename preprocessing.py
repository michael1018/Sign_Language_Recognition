import random
import cv2

from edge_detection import edge_detection
from erosion_and_dilation import erosion_and_dilation
from skin_seg_new import method4_Otsu

data_type = 'train'
path = f'./image/{data_type}/'
original_path = f'./original_photo/{data_type}/'


# 旋转
def rotate(image, scale=0.9):
    angle = random.randrange(-90, 90)  # 随机角度
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image


if __name__ == "__main__":
    for i in range(1, 11):
        cnt = 21  # 计数
        for j in range(1, 21):
            roi = cv2.imread(f'{original_path}{i}_{j}.jpg')
            # method1
            otsu = method4_Otsu(roi)
            res = erosion_and_dilation(otsu)

            # method2
            # res = edge_detection(roi)
            cv2.imwrite(path + f'{i}_{j}.jpg', res)
            for k in range(5):
                if cnt > 200:
                    continue
                img_rotation = rotate(res)  # 旋转
                cv2.imwrite(path + f'{i}_{cnt}.jpg', img_rotation)
                cnt += 1
                img_flip = cv2.flip(img_rotation, 1)  # 翻转
                cv2.imwrite(path + f'{i}_{cnt}.jpg', img_flip)
                cnt += 1
            print(i, '_', j, '完成')



