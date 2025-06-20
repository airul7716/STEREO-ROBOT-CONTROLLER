import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from handpose3d.utils import DLT, get_projection_matrix

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [720, 1280]

def estimate_finger_angles(landmarks_3d):
    # Uses landmarks: MCP (5, 9, 13, 17), PIP (6, 10, 14, 18), TIP (8, 12, 16, 20)
    finger_indices = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]  # Index to Pinky
    angles = []

    for a, b, c in finger_indices:
        if np.any(landmarks_3d[[a, b, c]] == -1):  # Skip if any are missing
            angles.append(0)
            continue
        vec1 = landmarks_3d[b] - landmarks_3d[a]
        vec2 = landmarks_3d[c] - landmarks_3d[b]
        angle_rad = np.arccos(np.clip(np.dot(vec1, vec2) /
                         (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        servo_angle = np.clip(180 - angle_deg, 0, 180)  # Invert to simulate bending
        angles.append(int(servo_angle))
    return angles

def draw_virtual_servos(image, angles):
    origin = (50, 100)
    spacing = 100
    for i, angle in enumerate(angles):
        center = (origin[0] + i * spacing, origin[1])
        cv.ellipse(image, center, (40, 40), 0, 0, angle, (0, 255, 0), 4)
        cv.putText(image, f"{angle}°", (center[0] - 20, center[1] + 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def run_mp(input_stream1, input_stream2, P0, P1):
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        if frame0.shape[1] != 720:
            frame0 = frame0[:, frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:, frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        frame0_rgb.flags.writeable = False
        frame1_rgb.flags.writeable = False
        results0 = hands0.process(frame0_rgb)
        results1 = hands1.process(frame1_rgb)

        frame0_keypoints, frame1_keypoints = [], []

        for results, frame, keypoints in zip([results0, results1], [frame0, frame1], [frame0_keypoints, frame1_keypoints]):
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for p in range(21):
                        pxl_x = int(round(frame.shape[1] * hand_landmarks.landmark[p].x))
                        pxl_y = int(round(frame.shape[0] * hand_landmarks.landmark[p].y))
                        keypoints.append([pxl_x, pxl_y])
            else:
                keypoints.extend([[-1, -1]] * 21)

        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                frame_p3ds.append([-1, -1, -1])
            else:
                frame_p3ds.append(DLT(P0, P1, uv1, uv2))

        frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        angles = estimate_finger_angles(frame_p3ds)

        frame0 = cv.cvtColor(frame0_rgb, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1_rgb, cv.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        servo_display = np.zeros((200, 600, 3), dtype=np.uint8)
        draw_virtual_servos(servo_display, angles)

        cv.imshow('cam0', frame0)
        cv.imshow('cam1', frame1)
        cv.imshow('Virtual Servos', servo_display)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

if __name__ == '__main__':
    input_stream1 = 1
    input_stream2 = 2
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)