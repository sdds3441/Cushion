import cv2
import numpy as np
import time

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
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 512, 512)

# 초기 색상 설정 (빨간색)
color = (0, 255, 0)

# 빨간색에서 파란색으로 변환하는 루프
for i in range(256):
    # 색상 계산
    current_color = (i, 0, 255 - i)

    # 원본 이미지를 복사하여 사용 (기존 이미지를 덮어쓰지 않기 위해)
    image_copy = image.copy()

    # 각 좌표에 점 찍기
    for point in points:
        cv2.circle(image_copy, point, radius=30, color=current_color, thickness=-1)

    # 이미지 보여주기
    cv2.imshow(window_name, image_copy)

    # 30ms 대기 (빠른 변화 시 조정 가능)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC 키를 누르면 루프 종료
        break

cv2.destroyAllWindows()
