import cv2

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start webcam
cap = cv2.VideoCapture(0)

# Frame counter for performance
frame_count = 0
detected_boxes = []

# Create fixed-size window
cv2.namedWindow("Smart Surveillance System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Surveillance System", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match window
    frame = cv2.resize(frame, (640, 480))

    # Get frame dimensions
    H, W = frame.shape[:2]

    # Define restricted area dynamically
    RX1 = int(W * 0.6)
    RY1 = int(H * 0.1)
    RX2 = int(W * 0.95)
    RY2 = int(H * 0.9)

    # ---------------- Image Filtering (Syllabus) ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---------------- Human Detection ----------------
    if frame_count % 5 == 0:
        boxes, _ = hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.03
        )
        detected_boxes = boxes

    frame_count += 1

    # ---------------- Draw Restricted Area ----------------
    cv2.rectangle(frame, (RX1, RY1), (RX2, RY2), (0, 0, 255), 2)
    cv2.putText(
        frame,
        "RESTRICTED AREA",
        (RX1, RY1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # ---------------- Draw Human Boxes ----------------
    for (x, y, w, h) in detected_boxes:
        # Padding for better coverage
        pad_w = int(0.15 * w)
        pad_h = int(0.25 * h)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(W, x + w + pad_w)
        y2 = min(H, y + h + pad_h)

        # Center of detected person
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Check restricted area
        if RX1 < cx < RX2 and RY1 < cy < RY2:
            color = (0, 0, 255)
            label = "HUMAN - RESTRICTED"
        else:
            color = (0, 255, 0)
            label = "HUMAN - NORMAL"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Show output
    cv2.imshow("Smart Surveillance System", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
