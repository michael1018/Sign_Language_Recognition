
import cv2
import numpy as np

from fourierDescriptor import fourierDesciptor

data_type = 'train'
path = './' + 'feature' + '/' + data_type + '/'
path_img = './' + 'image' + '/' + data_type + '/'

if __name__ == "__main__":
    for i in range(1, 11):
        for j in range(1, 201):
            roi = cv2.imread(path_img + str(i) + '_' + str(j) + '.jpg')

            ret, descirptor_in_use= fourierDesciptor(roi)
            descirptor_in_use = abs(descirptor_in_use)

            fd_name = path + str(i) + '_' + str(j) + '.txt'
            # fd_name = path + str(i) + '.txt'
            with open(fd_name, 'w', encoding='utf-8') as f:
                temp = descirptor_in_use[1]
                for k in range(1, len(descirptor_in_use)):
                    x_record = int(100 * descirptor_in_use[k] / temp)
                    f.write(str(x_record))
                    f.write(' ')
                f.write('\n')
            print(i, '_', j, 'complete')