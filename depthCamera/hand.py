import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream configuration

# Start the pipeline
pipeline.start(config)

try:
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while True:
            # Wait for a coherent frame from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert RealSense frame to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Convert BGR to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Check if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Example: Access specific landmark points
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        print(f"Landmark {id}: ({cx}, {cy})")

            else:
                print("No hands detected!")

            # Display the frame
            cv2.imshow("Hand Tracking with RealSense", frame)

            # Exit loop on 'Esc' key press
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    # Stop RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
