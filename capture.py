import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)

NUMBER_OF_PHOTOS = 20

data_type = 'test'
gesture = 1
new_dir_path = f'./original_photo/{data_type}'

photo_number = 5
width, height = 250, 250
x0, y0 = 30, 100
if not os.path.exists(new_dir_path):
    os.mkdir(new_dir_path)

print(f"Please make the gesture of {gesture} one and press s")
i = 1
while 1:
    ret, frame = cap.read()
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
    roi = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图
    capture_roi = frame[y0+1:y0 + height-1, x0+1:x0 + width-1]
    cv2.imshow("roi", roi)  # 显示手势框图
    if (cv2.waitKey(1) & 0xFF) == ord('s'):
        cv2.imwrite(f'{new_dir_path}/{gesture}_{i}.jpg', capture_roi)
        print(f"Save number:{i} photo")
        i += 1
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

    if i > photo_number:
        i = 1
        gesture += 1
        print(f"Please make the gesture of {gesture} one and press s")
    cv2.imshow("frame", frame)
cap.release()
cv2.destroyAllWindows()
