import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 Pose model (for limb tracking)
model = YOLO("yolov8m-pose.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

# Open video capture
cap = cv2.VideoCapture("../Data/ShoppingCCTVVideo.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO Pose estimation
    results = model(frame)

    # Extract keypoints for persons
    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class 0 = Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                score = float(box.conf[0])

                if score > 0.4:  # Ignore detections below 40% confidence
                    detections.append([[x1, y1, w, h], score])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Keypoints for limbs
        if result.keypoints:
            for kp in result.keypoints.xy:
                for point in kp:
                    x, y = map(int, point)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # Update tracker
    if detections:
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Person Tracking with Limb Movements", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
