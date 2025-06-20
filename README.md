# 🤖 Stereo Vision Hand Tracking and Robotic Hand Control

This project enables real-time 3D hand tracking using stereo cameras and MediaPipe, calculates joint angles, and controls a robotic hand via Arduino servos.

## 📦 Features
- Dual webcam-based stereo vision for 3D landmark triangulation
- MediaPipe-based 2D hand detection
- DLT triangulation for accurate 3D point estimation
- Joint angle estimation for MCP, PIP, DIP joints (and wrist roll)
- Servo control via Arduino using serial communication
- Virtual servo visualization for debugging

---

## 🖥️ Requirements

### 📌 Hardware
- Two USB webcams (camera indices 1 and 2 by default)
- Arduino Uno (or compatible)
- 6 servo motors (Thumb, Index, Middle, Ring, Pinky, Wrist)

### 🧰 Software
- Python 3.9+
- Required Python packages:
  ```bash
 - pip install opencv-python mediapipe pyserial numpy
