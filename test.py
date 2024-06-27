import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# CSV 파일 읽기
file_path = 'dataset/collected2.csv'  # 테스트 파일 경로를 지정하세요
data = pd.read_csv(file_path, header=None)

# 데이터 크기 확인
print(f"Total samples in test data: {data.shape[0]}")

# 입력 데이터와 레이블 분리
X_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values

# 데이터 표준화
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# 데이터 차원 조정 (CNN 입력 형태 맞춤)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# 모델 로드
model_path = 'models/my_cnn_model.h5'  # 모델 파일 경로를 지정하세요
model = tf.keras.models.load_model(model_path)

print(len(X_test_scaled))
# 예측 수행
predictions = model.predict(X_test_scaled)

predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
# 전체 정확도 계산
total_accuracy = np.mean(predicted_classes == y_test)
print(f'Total Accuracy: {total_accuracy:.4f}')
