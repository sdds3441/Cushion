import cv2
import numpy as np
import pandas as pd
import serial
import threading
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# .h5 모델 로드
model_path = 'models/my_cnn_model.h5'
model = tf.keras.models.load_model(model_path)

# 스케일러 로드 (학습 시 저장한 스케일러 사용)
scaler_path = 'models/scaler.pkl'
scaler = joblib.load(scaler_path)

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

            # 데이터 표준화 (학습 시 사용된 스케일러로 변환)
            test_val_scaled = scaler.transform(test_val)

            # 데이터 차원 조정 (CNN 입력 형태 맞춤)
            test_val_scaled = np.expand_dims(test_val_scaled, axis=2)

            # 모델 예측
            predictions = model.predict(test_val_scaled)
            predicted_class = np.argmax(predictions, axis=1)

            # 결과 콘솔에 출력
            print(f"Predicted class: {predicted_class[0]}")
            print(f"Latest data: {latest_data}")
            #print(f"Prediction probabilities: {predictions[0]}")
        else:
            print("No data received from Arduino yet.")

        time.sleep(1)  # 1초마다 예측 수행

# 예측을 수행하는 스레드 시작
threading.Thread(target=predict, daemon=True).start()

# 프로그램이 종료되지 않도록 유지
while True:
    time.sleep(1)
