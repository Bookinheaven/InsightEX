import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

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
                score = float(box.conf[0])
                detections.append([x1, y1, x2, y2, score])

    # Debugging: Check detection format
    print(f"Detections: {detections}")

    if detections:  # Ensure detections are not empty
        # Convert detections to the format expected by DeepSORT
        detections_list = []
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            w = x2 - x1
            h = y2 - y1
            detections_list.append([[x1, y1, w, h], confidence])  # Convert to (x, y, w, h)

        # Update DeepSORT tracker
        tracks = tracker.update_tracks(detections_list, frame=frame)

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