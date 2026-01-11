import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # force DirectShow backend
if not cap.isOpened():
    print("❌ Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        print("✅ Camera working, frame captured")
    else:
        print("❌ Camera opened but cannot read frame")
cap.release()
