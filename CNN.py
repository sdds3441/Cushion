import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CSV 파일 읽기
file_path = 'dataset/collected2.csv'  # 파일 경로를 지정하세요
data = pd.read_csv(file_path, header=None)

# 입력 데이터와 레이블 분리
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# CNN 모델 정의
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # 클래스 개수에 맞추어 출력층 설정
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 데이터 차원 조정 (CNN 입력 형태 맞춤)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 모델 훈련
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

model_save_path = 'models/my_cnn_model.h5'
model.save(model_save_path)
print(f'Model saved to {model_save_path}')