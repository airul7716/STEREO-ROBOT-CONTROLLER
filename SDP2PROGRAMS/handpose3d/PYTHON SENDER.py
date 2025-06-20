import serial
import time

# Adjust this to match your Arduino's COM port
SERIAL_PORT = 'COM5'  # Change to '/dev/ttyUSB0' or similar on Linux/Mac
BAUD_RATE = 9600

# Function to send servo angles to Arduino
def send_servo_angles(angles):
    # Clamp and format angles as comma-separated string
    angles = [str(min(max(int(a), 0), 180)) for a in angles]
    message = ','.join(angles) + '\n'
    print(f"Sending: {message.strip()}")

    ser.write(message.encode('utf-8'))

# Open serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset

    # Example loop: sending test values every second
    while True:
        # Example data - replace with values from your hand tracking system
        example_angles = [90, 170, 30, 25, 40, 10]  # Thumb to wrist
        send_servo_angles(example_angles)
        time.sleep(1)

except serial.SerialException as e:
    print(f"Serial error: {e}")

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
