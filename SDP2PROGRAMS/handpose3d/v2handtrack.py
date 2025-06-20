import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
import serial
import time
from utils import DLT, get_projection_matrix

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [720, 1280]

def estimate_finger_angles(landmarks_3d):
    finger_indices = [(2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    angles = []
    for a, b, c in finger_indices:
        if np.any(landmarks_3d[[a, b, c]] == -1):
            angles.append(0)
            continue
        vec1 = landmarks_3d[b] - landmarks_3d[a]
        vec2 = landmarks_3d[c] - landmarks_3d[b]
        angle_rad = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        servo_angle = np.clip(180 - angle_deg, 0, 180)
        angles.append(int(servo_angle))
    return angles

def estimate_wrist_roll(landmarks_3d):
    wrist = landmarks_3d[0]
    index_mcp = landmarks_3d[5]
    pinky_mcp = landmarks_3d[17]
    if np.any(wrist == -1) or np.any(index_mcp == -1) or np.any(pinky_mcp == -1):
        return 90
    vec1 = index_mcp - wrist
    vec2 = pinky_mcp - wrist
    normal = np.cross(vec1, vec2)
    ref = np.array([0, 0, 1])
    angle_rad = np.arccos(np.clip(np.dot(normal, ref) / np.linalg.norm(normal), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return int(np.clip(angle_deg, 0, 180))

def draw_virtual_servos(image, angles, wrist_angle, steps):
    origin = (50, 100)
    spacing = 100
    for i, (angle, step) in enumerate(zip(angles, steps[:5])):
        center = (origin[0] + i * spacing, origin[1])
        cv.ellipse(image, center, (40, 40), 0, 0, angle, (0, 255, 0), 4)
        cv.putText(image, f"{angle}°", (center[0] - 25, center[1] + 65),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv.putText(image, f"Step: {step}", (center[0] - 30, center[1] + 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    wrist_center = (origin[0] + 5 * spacing, origin[1])
    cv.ellipse(image, wrist_center, (40, 40), 0, 0, wrist_angle, (255, 0, 0), 4)
    cv.putText(image, f"Roll: {wrist_angle}°", (wrist_center[0] - 40, wrist_center[1] + 65),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.putText(image, f"Step: {steps[5]}", (wrist_center[0] - 30, wrist_center[1] + 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def to_step(value, reverse=False):
    step = int(np.clip(value / 10, 0, 17))
    return 17 - step if reverse else step

def to_thumb_step(angle):
    angle = np.clip(angle, 70, 170)
    return int((170 - angle) / 5)

def run_mp(input_stream1, input_stream2, P0, P1, ser):
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    for cap in [cap0, cap1]:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        results0 = hands0.process(frame0_rgb)
        results1 = hands1.process(frame1_rgb)

        keypoints0 = [[-1, -1]] * 21
        keypoints1 = [[-1, -1]] * 21

        if results0.multi_hand_landmarks and results0.multi_handedness:
            if results0.multi_handedness[0].classification[0].label == "Left":
                for p in range(21):
                    lm = results0.multi_hand_landmarks[0].landmark[p]
                    keypoints0[p] = [int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])]

        if results1.multi_hand_landmarks and results1.multi_handedness:
            if results1.multi_handedness[0].classification[0].label == "Left":
                for p in range(21):
                    lm = results1.multi_hand_landmarks[0].landmark[p]
                    keypoints1[p] = [int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])]

        landmarks_3d = []
        for uv1, uv2 in zip(keypoints0, keypoints1):
            if uv1[0] == -1 or uv2[0] == -1:
                landmarks_3d.append([-1, -1, -1])
            else:
                landmarks_3d.append(DLT(P0, P1, uv1, uv2))

        landmarks_3d = np.array(landmarks_3d)
        finger_angles = estimate_finger_angles(landmarks_3d)
        wrist_roll = estimate_wrist_roll(landmarks_3d)

        if all(a == 0 for a in finger_angles):
            steps = [9] * 6
        else:
            steps = [
                to_thumb_step(finger_angles[0]),          # ✅ Custom thumb mapping
                to_step(finger_angles[1], reverse=False),
                to_step(finger_angles[2], reverse=False),
                to_step(finger_angles[3], reverse=False),
                to_step(finger_angles[4], reverse=False),
                to_step(wrist_roll , reverse=False)
            ]

        try:
            data_line = ','.join(map(str, steps)) + '\n'
            ser.write(data_line.encode('utf-8'))
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            print(f"Sent: {data_line.strip()}")

        frame0_bgr = cv.cvtColor(frame0_rgb, cv.COLOR_RGB2BGR)
        frame1_bgr = cv.cvtColor(frame1_rgb, cv.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame0_bgr, results0.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        if results1.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame1_bgr, results1.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        servo_display = np.zeros((220, 600, 3), dtype=np.uint8)
        draw_virtual_servos(servo_display, finger_angles, wrist_roll, steps)

        cv.imshow('cam0', frame0_bgr)
        cv.imshow('cam1', frame1_bgr)
        cv.imshow('Virtual Servos', servo_display)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    input_stream1 = 1
    input_stream2 = 2
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    ser = serial.Serial('COM5', 9600)
    time.sleep(2)

    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    run_mp(input_stream1, input_stream2, P0, P1, ser)
