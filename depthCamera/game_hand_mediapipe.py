import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import math
import random
import time

# Mediapipe Hands 설정
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

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

# Mediapipe Hands 초기화
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 공과 손의 거리 계산 함수
def is_ball_grabbed(hand_landmarks, ball_position, ball_radius, frame_width, frame_height):
    # 손목과 손끝의 위치
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # 손목과 손끝 간의 거리 계산
    wrist_coords = (int(wrist.x * frame_width), int(wrist.y * frame_height))
    thumb_coords = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
    index_coords = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
    middle_coords = (int(middle_tip.x * frame_width), int(middle_tip.y * frame_height))
    ring_coords = (int(ring_tip.x * frame_width), int(ring_tip.y * frame_height))
    pinky_coords = (int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height))

    # 손바닥이 공을 잡았는지 확인
    # 공의 중심과 손목 및 손끝 사이의 거리 확인
    distances = []
    for finger_coords in [thumb_coords, index_coords, middle_coords, ring_coords, pinky_coords]:
        distance = math.sqrt((finger_coords[0] - ball_position[0]) ** 2 + (finger_coords[1] - ball_position[1]) ** 2)
        distances.append(distance)

    # 손끝이 공에 가까운지 확인
    if all(dist < (ball_radius + 50) for dist in distances):  # 손끝이 공에 가까운 경우
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

    # BGR -> RGB 변환 (Mediapipe는 RGB를 사용)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 추적
    results = hands.process(rgb_frame)

    # 공 색상 초기값
    ball_color = random.choice(ball_colors)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 공이 손에 잡혔는지 확인
            if is_ball_grabbed(hand_landmarks, ball_position, ball_radius, frame.shape[1], frame.shape[0]):
                # 주먹을 쥐었을 때 점수 증가
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                finger_tips = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                ]
                
                # 손목과 손끝 간의 거리를 계산하여 주먹을 쥐었는지 확인
                wrist_coords = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                distances = []
                for finger_tip in finger_tips:
                    finger_coords = (int(finger_tip.x * frame.shape[1]), int(finger_tip.y * frame.shape[0]))
                    distance = math.sqrt((finger_coords[0] - wrist_coords[0]) ** 2 + (finger_coords[1] - wrist_coords[1]) ** 2)
                    distances.append(distance)

                # 모든 손끝이 손목에 가까운 경우 주먹을 쥔 것으로 간주
                if all(distance < 150 for distance in distances):
                    score += 1
                    print(f"Score: {score} (Ball grabbed and fist closed)")

                    ball_visible = False  # 공 숨기기
                    print(f"Score: {score} (Ball grabbed and fist closed)")

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
