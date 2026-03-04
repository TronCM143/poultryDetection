import time
import cv2
from ultralytics import YOLO
from collections import deque

print("🔄 Starting advanced chicken counter...")

# =========================
# LOAD MODEL
# =========================
model_path = "runs/detect/train2/weights/best.pt"
print(f"📦 Loading model from: {model_path}")

try:
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# =========================
# OPEN CAMERA STREAM
# =========================
stream_url = "http://192.168.10.122:4747/video"
print(f"📡 Connecting to camera: {stream_url}")

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera stream")
    exit()
else:
    print("✅ Camera stream opened")

# =========================
# GET REAL RESOLUTION
# =========================
ret, test_frame = cap.read()
if ret:
    FRAME_WIDTH = int(test_frame.shape[1])
    FRAME_HEIGHT = int(test_frame.shape[0])
else:
    FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

print(f"📐 Camera resolution: {FRAME_WIDTH}×{FRAME_HEIGHT}")

# =========================
# CONFIGURATION
# =========================
RATE_WINDOW_SEC = 10.0
print(f"⏱️ Rate calculated over last {RATE_WINDOW_SEC} seconds")

# =========================
# TRACKING VARIABLES
# =========================
total_chickens = 0
counted_ids = set()
rate_deque = deque()
track_history = {}

# =========================
# CREATE WINDOW + TRACKBARS
# =========================
cv2.namedWindow("Chicken Counter")

def nothing(x):
    pass

start_left   = int(FRAME_WIDTH * 0.18)
start_right  = int(FRAME_WIDTH * 0.82)
start_top    = int(FRAME_HEIGHT * 0.15)
start_bottom = int(FRAME_HEIGHT * 0.85)

cv2.createTrackbar('Left',   'Chicken Counter', start_left,   FRAME_WIDTH,  nothing)
cv2.createTrackbar('Right',  'Chicken Counter', start_right,  FRAME_WIDTH,  nothing)
cv2.createTrackbar('Top',    'Chicken Counter', start_top,    FRAME_HEIGHT, nothing)
cv2.createTrackbar('Bottom', 'Chicken Counter', start_bottom, FRAME_HEIGHT, nothing)

print("🎛️ Adjust ROI with trackbars")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ----- GET ROI -----
    roi_x1 = cv2.getTrackbarPos('Left', 'Chicken Counter')
    roi_x2 = cv2.getTrackbarPos('Right', 'Chicken Counter')
    roi_y1 = cv2.getTrackbarPos('Top', 'Chicken Counter')
    roi_y2 = cv2.getTrackbarPos('Bottom', 'Chicken Counter')

    if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
        continue

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # ----- RUN TRACKING -----
    results = model.track(
        roi,
        persist=True,
        conf=0.35,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,255,0), 2)

    # ----- PROCESS DETECTIONS -----
    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box in results[0].boxes:

            track_id = int(box.id)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to full-frame coordinates
            x1 += roi_x1
            x2 += roi_x1
            y1 += roi_y1
            y2 += roi_y1

            center_x = int((x1 + x2) / 2)

            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0,255,0), 2)

            if track_id not in track_history:
                track_history[track_id] = center_x
                continue

            prev_x = track_history[track_id]

            width = roi_x2 - roi_x1
            right_zone = roi_x1 + int(width * 0.80)

            # ENTER from RIGHT → moving LEFT
            if prev_x > center_x and prev_x > right_zone:
                if track_id not in counted_ids:
                    total_chickens += 1
                    counted_ids.add(track_id)
                    rate_deque.append(current_time)

            track_history[track_id] = center_x

    # ----- CLEAN RATE WINDOW -----
    while rate_deque and rate_deque[0] < current_time - RATE_WINDOW_SEC:
        rate_deque.popleft()

    chickens_in_last_window = len(rate_deque)

    rate_per_minute = (
        chickens_in_last_window * (60.0 / RATE_WINDOW_SEC)
        if RATE_WINDOW_SEC > 0 else 0
    )

    # ----- DRAW INFO -----
    cv2.putText(frame, f"TOTAL: {total_chickens}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,255,0), 3)

    cv2.putText(frame, f"RATE: {rate_per_minute:.1f} per min",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0,255,255), 2)

    cv2.imshow("Chicken Counter", frame)

    # ----- CONTROLS -----
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