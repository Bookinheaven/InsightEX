import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")
"""
yolov8n.pt → Nano
yolov8s.pt → Small
yolov8m.pt → Medium
yolov8l.pt → Large
yolov8x.pt → Extra Large
"""
# Open a video file or webcam (use 0 for webcam)
cap = cv2.VideoCapture("../Data/ShoppingCCTVVideo.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame)
    print(results)
    # Count number of people detected
    people_count = sum(1 for r in results[0].boxes if r.cls == 0)  # Class 0 = Person

    # Draw bounding boxes
    for box in results[0].boxes:
        if box.cls == 0:  # Only draw for 'person' class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f"People Count: {people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show output
    cv2.imshow("YOLOv8 People Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
