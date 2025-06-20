import serial
import time
import threading

class SerialComm:
    def __init__(self, port='COM5', baudrate=115200, timeout=1, send_rate_hz=10):
        """
        Initialize the serial communication interface.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.send_interval = 1.0 / send_rate_hz
        self.last_send_time = 0

        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Give Arduino time to reset
            print(f"[SerialComm] Connected to {port} at {baudrate} baud.")
        except serial.SerialException as e:
            print(f"[SerialComm] Failed to connect to {port}: {e}")
            self.ser = None

        self.lock = threading.Lock()

    def send_angles(self, angles):
        """
        Send a list of 6 servo angles over serial.
        Args:
            angles: List of 6 integers (0-180 degrees).
        """
        if self.ser is None or not self.ser.is_open:
            return

        if len(angles) != 6:
            print("[SerialComm] Error: Expected 6 angles, got", len(angles))
            return

        now = time.time()
        if now - self.last_send_time < self.send_interval:
            return  # throttle sending rate

        message = ','.join(str(int(a)) for a in angles) + '\n'
        with self.lock:
            try:
                self.ser.write(message.encode('utf-8'))
                self.last_send_time = now
                print(f"[SerialComm] Sent: {message.strip()}")
            except Exception as e:
                print(f"[SerialComm] Write error: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SerialComm] Serial connection closed.")
