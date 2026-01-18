import os
import time
import cv2
import string

# ===============================
# CONFIGURATION
# ===============================
DATA_DIR = "./data2"
ALPHABETS = list(string.ascii_uppercase)  # A–Z
IMAGES_PER_ALPHABET = 400
CAPTURE_DELAY = 0.08   # ~12.5 FPS

# ===============================
# SETUP DATA DIRECTORY
# ===============================
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# OPEN WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("✅ Webcam opened successfully")

# ===============================
# DATA COLLECTION LOOP
# ===============================
for letter in ALPHABETS:
    class_path = os.path.join(DATA_DIR, letter)
    os.makedirs(class_path, exist_ok=True)

    print(f"\n📸 Ready for alphabet: {letter}")
    print('👉 Show the sign and press "Q" to start capturing')

    # -------- WAIT FOR Q --------
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame,
            f"Alphabet {letter}",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )
        cv2.putText(
            frame,
            'Press "Q" to start capture',
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Dataset Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"▶ Capturing images for {letter}...")

    # -------- CAPTURE FRAMES --------
    counter = 0
    while counter < IMAGES_PER_ALPHABET:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Dataset Collection", frame)

        file_path = os.path.join(class_path, f"{counter}.jpg")
        cv2.imwrite(file_path, frame)

        counter += 1
        time.sleep(CAPTURE_DELAY)

        # Optional emergency stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"✅ Finished {letter}: {counter} images saved")

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("\n🎉 Dataset collection completed successfully for A–Z")
