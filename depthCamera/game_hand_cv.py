import cv2
import numpy as np
import pyrealsense2 as rs
import math
import random
import time

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 색상 스트림 설정
pipeline.start(config)

# 공 설정
ball_radius = 20
ball_color = (0, 0, 255)  # 빨간색
ball_speed = 5
ball_position = [320, 240]  # 화면 중앙
ball_direction = [ball_speed, ball_speed]
# 공의 색상 목록
ball_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)]  # 빨간, 초록, 파랑, 노랑, 하늘색

# 점수 설정
score = 0
ball_visible = True  # 공의 가시성

# 손 인식에 사용할 함수들
def get_hand_contours(frame):
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 블러 적용하여 노이즈 감소
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 손 모양을 추출하기 위한 이진화 (흑백 변환)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # 이진화된 이미지에서 외곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 공과 손의 거리 계산 함수
def is_ball_grabbed(hand_contours, ball_position, ball_radius):
    for contour in hand_contours:
        # 손의 외곽선에 맞는 최소한의 경계 사각형을 구함
        x, y, w, h = cv2.boundingRect(contour)
        
        # 손의 영역 내에서 공이 있는지 체크
        hand_center = (x + w // 2, y + h // 2)
        distance = math.sqrt((hand_center[0] - ball_position[0]) ** 2 + (hand_center[1] - ball_position[1]) ** 2)
        
        if distance < ball_radius + 50:  # 손의 중앙과 공의 거리 비교
            return True
    return False

# 게임 루프
while True:
    # RealSense 프레임 받아오기
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # RealSense 색상 프레임을 NumPy 배열로 변환
    frame = np.asanyarray(color_frame.get_data())

    # 손 외곽선 찾기
    hand_contours = get_hand_contours(frame)

    # 공 색상 초기값
    ball_color = random.choice(ball_colors)

    if hand_contours:
        # 손에 잡혔는지 확인
        if is_ball_grabbed(hand_contours, ball_position, ball_radius):
            # 주먹을 쥐었을 때 점수 증가
            score += 1
            print(f"Score: {score} (Ball grabbed)")

            ball_visible = False  # 공 숨기기

            # 1초 동안 공을 숨기고 다른 위치에서 공을 다시 나타나게 하기
            time.sleep(1)
            ball_visible = True  # 공을 다시 보이게 함
            ball_position = [random.randint(100, frame.shape[1] - 100), random.randint(100, frame.shape[0] - 100)]
            ball_color = random.choice(ball_colors)  # 새로운 색상 선택

    # 공의 위치 업데이트
    ball_position[0] += ball_direction[0]
    ball_position[1] += ball_direction[1]

    # 공이 화면 경계를 넘지 않도록 처리
    if ball_position[0] - ball_radius <= 0 or ball_position[0] + ball_radius >= frame.shape[1]:
        ball_direction[0] *= -1
    if ball_position[1] - ball_radius <= 0 or ball_position[1] + ball_radius >= frame.shape[0]:
        ball_direction[1] *= -1

    # 공 그리기
    if ball_visible:
        cv2.circle(frame, (ball_position[0], ball_position[1]), ball_radius, ball_color, -1)

    # 화면에 점수 표시
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 프레임 표시
    cv2.imshow("Hand Tracking Game", frame)

    # 'ESC' 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# RealSense 파이프라인 종료
pipeline.stop()
cv2.destroyAllWindows()
