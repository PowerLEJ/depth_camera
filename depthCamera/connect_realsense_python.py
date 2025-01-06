import pyrealsense2 as rs  # Intel RealSense SDK 모듈
import numpy as np         # 행렬 및 배열 계산을 위한 라이브러리
import cv2                 # OpenCV 라이브러리, 이미지 처리 및 GUI 제공

# RealSense 파이프라인 초기화
pipe = rs.pipeline()       # RealSense 데이터를 관리할 파이프라인 생성
cfg = rs.config()          # 스트림 설정을 위한 구성 객체 생성

# 컬러 스트림 설정 (해상도: 640x480, 형식: BGR, FPS: 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 깊이 스트림 설정 (해상도: 640x480, 형식: Z16, FPS: 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 설정한 스트림으로 파이프라인 시작
pipe.start(cfg)

# 초기 포인트 설정 (화면 중앙 좌표)
point = (400, 300)

# 마우스 이벤트 콜백 함수
def show_distance(event, x, y, args, params):
    """
    마우스 클릭 시 클릭한 위치 좌표를 업데이트
    """
    global point
    point = (x, y)  # 현재 마우스 좌표로 업데이트

# OpenCV 창 생성 및 마우스 콜백 함수 등록
cv2.namedWindow("rgb frame")  # 컬러 이미지를 표시할 창 생성
cv2.setMouseCallback("rgb frame", show_distance)  # "rgb frame" 창에서 마우스 이벤트 처리

# 메인 루프
while True:
    # 프레임 가져오기 (깊이 프레임과 컬러 프레임)
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    # 깊이 및 컬러 데이터를 NumPy 배열로 변환
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 깊이 이미지를 시각화하기 위해 컬러맵 적용
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    # 현재 포인트에 빨간 원 그리기
    cv2.circle(color_image, point, 4, (0, 0, 255))  # 포인트 좌표에 원 표시
    print(point)  # 현재 포인트 좌표 출력

    # 깊이 값 가져오기 (현재 포인트의 깊이값)
    distance = depth_image[point[1], point[0]]  # y, x 순서로 접근
    print(distance)  # 깊이값 출력 (밀리미터 단위)

    # 깊이값을 컬러 이미지에 텍스트로 표시
    cv2.putText(
        color_image,
        "{}mm".format(distance),  # 깊이값(mm 단위) 표시
        (point[0], point[1]),    # 텍스트 위치 (현재 포인트 좌표)
        cv2.FONT_HERSHEY_PLAIN,  # 폰트 스타일
        2,                       # 폰트 크기
        (0, 0, 0),               # 텍스트 색상 (검정색)
        2                        # 텍스트 두께
    )

    # 컬러 이미지 및 깊이 이미지 표시
    cv2.imshow('rgb frame', color_image)  # 컬러 이미지 표시
    cv2.imshow('depth frame', depth_cm)  # 컬러맵이 적용된 깊이 이미지 표시

    # 키 입력 대기: 'q'를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 파이프라인 정리 및 리소스 해제
pipe.stop()