import cv2

def check_cameras(max_index=5):
    print("Checking available camera indices:")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            print(f"✅ Camera available at index {i}")
        else:
            print(f"❌ No camera at index {i}")
        cap.release()

check_cameras()
