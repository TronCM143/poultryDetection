import time
import cv2
import torch
from ultralytics import YOLO
from collections import deque

print("🔄 Starting chicken counter...")

# =========================
# LOAD MODEL
# =========================
model_path = "runs/detect/train2/weights/best.pt"
#model_path = "yolov8s.pt"
try:
    model = YOLO(model_path)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"✅ Model loaded on {device}")

except Exception as e:
    print("❌ Error loading model:", e)
    exit()


# =========================
# OPEN CAMERA STREAM
# =========================
stream_url = "http://192.168.1.21:4747/video"
print(f"📡 Connecting to camera: {stream_url}")

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ ERROR: Could not open video")
    exit()

print("✅ Camera connected")


# =========================
# TRACKING VARIABLES
# =========================
total_chickens = 0
counted_ids = set()
track_history = {}

rate_deque = deque()
RATE_WINDOW_SEC = 10


# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    frame_height, frame_width = frame.shape[:2]

    # =========================
    # COUNTING LINE (CENTER)
    # =========================
    line_x = frame_width // 2

    cv2.line(
        frame,
        (line_x, 0),
        (line_x, frame_height),
        (0, 0, 255),
        3
    )

    # =========================
    # YOLO TRACKING
    # =========================
    results = model.track(
        frame,
        persist=True,
        conf=0.30,     # 40% confidence threshold
        imgsz=512,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # =========================
    # PROCESS DETECTIONS
    # =========================
    if results[0].boxes is not None and results[0].boxes.id is not None:

        for box in results[0].boxes:

            confidence = float(box.conf)

            # Ignore low confidence detections
            if confidence < 0.40:
                continue

            track_id = int(box.id)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center_x = int((x1 + x2) / 2)

            # Draw detection box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # Save first position
            if track_id not in track_history:
                track_history[track_id] = center_x
                continue

            prev_x = track_history[track_id]

            # =========================
            # LINE CROSSING LOGIC
            # =========================
            if prev_x > line_x and center_x <= line_x:

                if track_id not in counted_ids:

                    total_chickens += 1
                    counted_ids.add(track_id)
                    rate_deque.append(current_time)

            track_history[track_id] = center_x


    # =========================
    # CLEAN RATE WINDOW
    # =========================
    while rate_deque and rate_deque[0] < current_time - RATE_WINDOW_SEC:
        rate_deque.popleft()


    # =========================
    # DISPLAY COUNTER
    # =========================
    cv2.putText(
        frame,
        f"TOTAL: {total_chickens}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )


    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("Chicken Counter", frame)


    key = cv2.waitKey(1)

    if key == 27:
        print("🛑 ESC pressed, exiting...")
        break

    elif key in [ord('r'), ord('R')]:
        counted_ids.clear()
        rate_deque.clear()
        total_chickens = 0
        print("✅ Counters reset")


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

print(f"👋 Program finished. Total chickens counted: {total_chickens}")


# import cv2
# from ultralytics import YOLO

# # =========================
# # LOAD MODEL
# # =========================
# model_path = "runs/detect/train2/weights/best.pt"
# model = YOLO(model_path)

# # =========================
# # LOAD IMAGE
# # =========================
# image_path = "24488.jpg"

# frame = cv2.imread(image_path)

# if frame is None:
#     print("❌ Could not load image")
#     exit()

# # =========================
# # RUN DETECTION
# # =========================
# results = model(
#     frame,
#     conf=0.15,
#     imgsz=960
# )

# # =========================
# # COUNT CHICKENS
# # =========================
# chicken_count = 0

# for box in results[0].boxes:

#     x1, y1, x2, y2 = box.xyxy[0].tolist()
#     conf = float(box.conf)

#     chicken_count += 1

#     # draw box
#     cv2.rectangle(
#         frame,
#         (int(x1), int(y1)),
#         (int(x2), int(y2)),
#         (0,255,0),
#         2
#     )

#     cv2.putText(
#         frame,
#         f"{conf:.2f}",
#         (int(x1), int(y1)-10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0,255,0),
#         2
#     )

# # =========================
# # DISPLAY RESULT
# # =========================
# cv2.putText(
#     frame,
#     f"Chickens: {chicken_count}",
#     (20,60),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     1.2,
#     (0,255,0),
#     3
# )

# print(f"🐔 Chickens detected: {chicken_count}")

# cv2.imshow("Chicken Detection", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()