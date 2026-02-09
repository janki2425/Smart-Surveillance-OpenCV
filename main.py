import cv2
import os
import winsound
from datetime import datetime
from ultralytics import YOLO

# ---------------- Setup ----------------
model = YOLO("yolov8n.pt")  # nano model

cap = cv2.VideoCapture(0)

# Alert control
alert_triggered = False
ALERT_COOLDOWN_FRAMES = 30
alert_counter = 0

# Create snapshots folder
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# Window
cv2.namedWindow("Smart Surveillance System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Surveillance System", 640, 480)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    H, W = frame.shape[:2]

    # -------- Restricted Area --------
    RX1 = int(W * 0.6)
    RY1 = int(H * 0.1)
    RX2 = int(W * 0.95)
    RY2 = int(H * 0.9)

    # Draw restricted area
    cv2.rectangle(frame, (RX1, RY1), (RX2, RY2), (0, 0, 255), 2)
    cv2.putText(
        frame, "RESTRICTED AREA",
        (RX1, RY1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 255), 2
    )

    restricted_detected = False

    # -------- YOLO Detection --------
    results = model(frame, stream=True, conf=0.5)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if RX1 < cx < RX2 and RY1 < cy < RY2:
                color = (0, 0, 255)
                label = "HUMAN - RESTRICTED"
                restricted_detected = True
            else:
                color = (0, 255, 0)
                label = "HUMAN - NORMAL"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2
            )

    # -------- Alert + Snapshot --------
    if restricted_detected and not alert_triggered:
        winsound.Beep(1500, 600)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"snapshots/intruder_{timestamp}.jpg" #saves photo in snapshots folder
        cv2.imwrite(filename, frame)

        alert_triggered = True
        alert_counter = 0

    if alert_triggered:
        alert_counter += 1
        if alert_counter > ALERT_COOLDOWN_FRAMES:
            alert_triggered = False

    # -------- Display --------
    cv2.imshow("Smart Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
