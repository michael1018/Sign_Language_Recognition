import cv2

from erosion_and_dilation import erosion_and_dilation
from fourierDescriptor import fourierDesciptor
from skin_seg_new import method4_Otsu


def get_feature(roi):
    ret, descirptor_in_use = fourierDesciptor(roi)
    descirptor_in_use = abs(descirptor_in_use)
    temp = descirptor_in_use[1]
    feature = []
    for k in range(1, len(descirptor_in_use)):
        x_record = int(100 * descirptor_in_use[k] / temp)
        feature.append(x_record)
    return feature


if __name__ == '__main__':
    roi = cv2.imread(f'./image/test/2_1.jpg')
    otsu = method4_Otsu(roi)
    res = erosion_and_dilation(otsu)
    feature = get_feature(roi)
    print(feature)
