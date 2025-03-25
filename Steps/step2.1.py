import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model trained on store shelves
model_shelf = YOLO("yolov8s-shelves.pt")  # Custom-trained model for shelves
model_person = YOLO("yolov8m.pt")  # Person detection model

# Initialize DeepSORT for person tracking
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

# Open video capture
cap = cv2.VideoCapture("../Data/ShoppingCCTVVideo.mp4")

# Store detected shelves dynamically
shelves = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect shelves in the frame
    shelf_results = model_shelf(frame)

    # Extract shelf bounding boxes
    shelves.clear()
    for result in shelf_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            shelves[len(shelves) + 1] = (x1, y1, x2, y2)  # Store shelf dynamically

    # Detect persons
    results = model_person(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class 0 = Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                score = float(box.conf[0])

                if score > 0.4:  # Ignore low confidence
                    detections.append([[x1, y1, w, h], score])

    # Track persons
    if detections:
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Find which shelf the person is closest to
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                focus_shelf = "None"

                for shelf_id, (sx1, sy1, sx2, sy2) in shelves.items():
                    if sx1 < person_center[0] < sx2 and sy1 < person_center[1] < sy2:
                        focus_shelf = f"Shelf {shelf_id}"
                        break

                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id} ({focus_shelf})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw detected shelves
    for shelf_id, (sx1, sy1, sx2, sy2) in shelves.items():
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
        cv2.putText(frame, f"Shelf {shelf_id}", (sx1, sy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Shelf & Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
