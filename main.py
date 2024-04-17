import cv2
import joblib
import mediapipe as mp
import numpy as np

from edge_detection import edge_detection
from erosion_and_dilation import erosion_and_dilation
from get_feature import get_feature
from skin_seg_new import method4_Otsu

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv2.LINE_AA  # 印出文字的邊框
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

photo_number = 5
width, height = 250, 250
x0, y0 = 30, 150

model_path = 'model/RandomForest.pkl'
model = joblib.load(model_path)

i = 1
predict_numbers = []
text = None
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while 1:
        ret, frame = cap.read()
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
        roi = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图
        capture_roi = frame[y0 + 1:y0 + height - 1, x0 + 1:x0 + width - 1]
        cv2.imshow("roi", roi)  # 显示手势框图
        results = hands.process(roi)  # 偵測手勢
        if results.multi_hand_landmarks:
            res = edge_detection(roi)
            feature = get_feature(res)
            y_predict = model.predict([feature])
            if y_predict is not None:
                predict_numbers.append(y_predict[0])
            if len(predict_numbers) > 3:
                print(predict_numbers)
                number = max(predict_numbers,key=predict_numbers.count)
                text = str(number)
                predict_numbers = []
            cv2.putText(frame, text, (30, 120), fontFace, 4, (255, 255, 255), 10, lineType)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        cv2.imshow("frame", frame)
cap.release()
cv2.destroyAllWindows()
