import cv2
from ultralytics import YOLO

print("🔄 Starting chicken counter...")

# === LOAD MODEL ===
model_path = "chickenCounting.v1i.yolov8/runs/detect/train/weights/best.pt"
print(f"📦 Loading model from: {model_path}")

try:
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# === OPEN CAMERA STREAM ===
stream_url = "http://192.168.0.102:4747/video"
print(f"📡 Connecting to camera: {stream_url}")

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera stream")
    exit()
else:
    print("✅ Camera stream opened")

# === MAIN LOOP ===
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame from camera")
        break

    frame_count += 1
    print(f"📷 Frame {frame_count} received")

    try:
        results = model(frame)
    except Exception as e:
        print("❌ Inference error:", e)
        break

    # count chickens
    boxes = results[0].boxes
    chicken_count = len(boxes)

    print(f"🐔 Detected chickens: {chicken_count}")

    # draw boxes
    annotated = results[0].plot()

    # display count
    cv2.putText(
        annotated,
        f"Chickens: {chicken_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Chicken Counter", annotated)

    # press ESC to exit
    if cv2.waitKey(1) == 27:
        print("🛑 ESC pressed, exiting...")
        break

cap.release()
cv2.destroyAllWindows()

print("👋 Program finished")