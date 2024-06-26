import cv2
import numpy as np
import pandas as pd
import serial
import threading
import time

# KNN 객체 생성
knn = cv2.ml.KNearest_create()

# 학습 데이터 읽기
file = np.genfromtxt('dataset//collected.csv', delimiter=',')
fsr_val = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

# KNN 모델 학습
knn.train(fsr_val, cv2.ml.ROW_SAMPLE, label)

# 아두이노와 시리얼 통신 설정
ser = serial.Serial('COM8', 9600)
time.sleep(2)  # 시리얼 통신이 안정되도록 잠시 대기

# 최신 데이터를 저장할 변수
latest_data = None


def read_from_arduino():
    global latest_data
    while True:
        if ser.in_waiting > 0:
            arduino_data = ser.readline().decode('utf-8').strip()
            latest_data = list(map(float, arduino_data.split(',')))


# 시리얼 포트에서 데이터를 읽는 스레드 시작
threading.Thread(target=read_from_arduino, daemon=True).start()


def predict():
    global latest_data
    while True:
        if latest_data is not None:
            # 최신 데이터를 numpy 배열로 변환하고 2차원 배열로 reshape
            test_val = np.array(latest_data).reshape(1, -1).astype(np.float32)

            # KNN 예측
            ret, results, neighbours, dist = knn.findNearest(test_val, 3)

            # 결과 콘솔에 출력
            print(f"Results: {results}")
            print(f"Neighbours: {neighbours}")
            print(f"Distances: {dist}")
        else:
            print("No data received from Arduino yet.")

        time.sleep(1)  # 1초마다 예측 수행


# 예측을 수행하는 스레드 시작
threading.Thread(target=predict, daemon=True).start()

# 프로그램이 종료되지 않도록 유지
while True:
    time.sleep(1)
