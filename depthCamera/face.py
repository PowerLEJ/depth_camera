import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Mediapipe Face Mesh 초기화
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# RealSense 카메라 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream 설정

# RealSense 파이프라인 시작
pipeline.start(config)

while True:
    # RealSense 프레임 받아오기
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # RealSense 색상 프레임을 numpy 배열로 변환
    frame = np.asanyarray(color_frame.get_data())

    # BGR -> RGB 변환 (Mediapipe는 RGB를 사용)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 및 랜드마크 인식
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # 얼굴 인식된 프레임 출력
    cv2.imshow("Face Mesh", frame)

    # 'ESC' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# RealSense 파이프라인 종료
pipeline.stop()
cv2.destroyAllWindows()
