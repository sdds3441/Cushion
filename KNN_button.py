import cv2
import numpy as np
import pandas as pd
import tkinter as tk
import serial
import threading

# KNN 객체 생성
knn = cv2.ml.KNearest_create()

# 학습 데이터 읽기
file = np.genfromtxt('dataset//collected.csv', delimiter=',')
fsr_val = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

# KNN 모델 학습
knn.train(fsr_val, cv2.ml.ROW_SAMPLE, label)

# 아두이노와 시리얼 통신 설정
# 시리얼 포트와 보드레이트는 아두이노 설정에 맞게 변경해야 합니다.
ser = serial.Serial('COM8', 9600)

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


# Tkinter GUI 설정
root = tk.Tk()
root.title("KNN Prediction")

# 버튼 생성
predict_button = tk.Button(root, text="Predict using KNN", command=predict)
predict_button.pack(pady=10)

# GUI 실행
root.mainloop()

# 시리얼 통신 종료
ser.close()
