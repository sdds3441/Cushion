import cv2
import numpy as np
import serial
import threading
import time

# KNN 객체 생성
knn = cv2.ml.KNearest_create()

# 학습 데이터 읽기
file = np.genfromtxt('dataset/collected.csv', delimiter=',')
fsr_val = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

# KNN 모델 학습
knn.train(fsr_val, cv2.ml.ROW_SAMPLE, label)

# 아두이노와 시리얼 통신 설정
ser = serial.Serial('COM8', 9600)
time.sleep(2)  # 시리얼 통신이 안정되도록 잠시 대기

# 최신 데이터를 저장할 변수
latest_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 이미지 불러오기
image_path = 'dataset/cushion2.png'  # 여기에 이미지 파일 경로를 입력하세요
image = cv2.imread(image_path)

# 이미지가 제대로 불러와졌는지 확인
if image is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

# 점을 찍을 좌표 설정 (예: (x, y) = (100, 150))
height, width, _ = image.shape
qheight = height // 4
qwidth = width // 4
points = [(qwidth * 2, qheight // 2), (qwidth // 2, qheight * 2), (qwidth * 3 + qwidth // 2, qheight * 2),
          (qwidth * 2, qheight * 3 + qheight // 2)]

# 창 이름 설정
window_name = 'Image with Changing Dots'


# 창 생성 및 크기 설정 (예: 800x600)
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name, 512, 512)


def read_from_arduino():
    global latest_data
    while True:
        if ser.in_waiting > 0:
            arduino_data = ser.readline().decode('utf-8').strip()
            latest_data = list(map(float, arduino_data.split(',')))


def predict_color():
    global latest_data
    while True:
        if latest_data is not None:
            # 최신 데이터를 numpy 배열로 변환하고 2차원 배열로 reshape
            test_val = np.array(latest_data).reshape(1, -1).astype(np.float32)

            # KNN 예측
            ret, results, neighbours, dist = knn.findNearest(test_val, 3)

            # 결과 값 출력
            print(f"Results: {results}")

            # 결과 값에 따라 색상 변경 (작을수록 빨간색, 클수록 파란색)


def cal_color(direction):
    mean = np.mean(direction)
    if 750 < mean:
        mean = 750

    normalized = mean / 750
    color = (0, int(255 * (1 - normalized)), int(255 * normalized))
    return color


def display_image():
    while True:
        global latest_data

        left = [latest_data[1], latest_data[2], latest_data[5], latest_data[6]]
        right = [latest_data[0], latest_data[3], latest_data[4], latest_data[7]]
        forward = [latest_data[0], latest_data[1], latest_data[4], latest_data[5]]
        backward = [latest_data[2], latest_data[3], latest_data[6], latest_data[7]]

        left_color = cal_color(left)
        right_color = cal_color(right)
        forward_color = cal_color(forward)
        backward_color = cal_color(backward)

        # 원본 이미지를 복사하여 사용 (기존 이미지를 덮어쓰지 않기 위해)
        image_copy = image.copy()

        # 각 좌표에 점 찍기
        cv2.circle(image_copy, points[0], radius=30, color=forward_color, thickness=-1)
        cv2.circle(image_copy, points[1], radius=30, color=left_color, thickness=-1)
        cv2.circle(image_copy, points[2], radius=30, color=right_color, thickness=-1)
        cv2.circle(image_copy, points[3], radius=30, color=backward_color, thickness=-1)

        # 이미지 보여주기
        cv2.imshow("window_name", image_copy)

        # 30ms 대기
        key = cv2.waitKey(30)
        if key == 27:  # ESC 키를 누르면 루프 종료
            break


# 시리얼 포트에서 데이터를 읽는 스레드 시작
threading.Thread(target=read_from_arduino, daemon=True).start()

# 예측을 수행하는 스레드 시작
threading.Thread(target=predict_color, daemon=True).start()

# 이미지를 표시하는 스레드 시작
threading.Thread(target=display_image, daemon=True).start()

# 프로그램이 종료되지 않도록 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("프로그램 종료")
finally:
    ser.close()
    cv2.destroyAllWindows()
