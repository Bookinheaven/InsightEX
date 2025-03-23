import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize DeepSORT tracker with improved parameters
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

# Open video capture
cap = cv2.VideoCapture("../Data/ShoppingCCTVVideo.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection
    results = model(frame)

    # Extract bounding boxes for persons (class 0)
    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class 0 = Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                score = float(box.conf[0])

                if score > 0.4:  # Ignore detections below 40% confidence
                    detections.append([[x1, y1, w, h], score])

    # Debugging: Check detection format
    print(f"Detections: {detections}")

    if detections:
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and track IDs
        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("DeepSORT Person Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
