import joblib
import cv2

from edge_detection import edge_detection
from erosion_and_dilation import erosion_and_dilation
from get_feature import get_feature
from skin_seg_new import method4_Otsu


def test_model_method1():
    roi = cv2.imread(f'./original_photo/test/2_1.jpg')
    model = joblib.load('./model/RandomForest.pkl')
    otsu = method4_Otsu(roi)
    res = erosion_and_dilation(otsu)
    feature = get_feature(res)
    y_predict = model.predict([feature])
    print(y_predict)


def test_model_method2():
    roi = cv2.imread(f'./original_photo/test/1_1.jpg')
    model = joblib.load('./model/RandomForest_edge.pkl')
    res = edge_detection(roi, show=True)
    feature = get_feature(res)
    y_predict = model.predict([feature])
    print(y_predict)


if __name__ == '__main__':
    # test_model_method1()
    test_model_method2()
    cv2.waitKey(0)
