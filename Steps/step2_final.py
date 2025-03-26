import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from openvino import Core

# Load YOLOv8 Pose model with OpenVINO
ie = Core()
model = YOLO("yolov8m-pose.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

# Open video file
cap = cv2.VideoCapture("../Data/ShoppingCCTVVideo.mp4")

# Heatmap storage
heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)

# Entry zones storage
entry_zones = []
drawing = False
start_x, start_y = -1, -1

# Mouse callback function
def draw_entry_zone(event, x, y, flags, param):
    global start_x, start_y, drawing, entry_zones

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        entry_zones.append((start_x, start_y, x, y))
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove entry zones if right-clicked inside
        entry_zones = [(ex1, ey1, ex2, ey2) for (ex1, ey1, ex2, ey2) in entry_zones if not (ex1 <= x <= ex2 and ey1 <= y <= ey2)]

# Function to check if a bounding box is in an entry zone
def is_in_entry_zone(x1, y1, x2, y2):
    for (ex1, ey1, ex2, ey2) in entry_zones:
        if x1 >= ex1 and x2 <= ex2 and y1 >= ey1 and y2 <= ey2:
            return True
    return False

cv2.namedWindow("Video Stream")
cv2.setMouseCallback("Video Stream", draw_entry_zone)

prev_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                score = float(box.conf[0])
                if score > 0.4 and not is_in_entry_zone(x1, y1, x2, y2):
                    detections.append([[x1, y1, w, h], score])
                    heatmap[y1:y2, x1:x2] += 1

                # Draw keypoints for limbs
                if result.keypoints:
                    for kp in result.keypoints.xy:
                        for point in kp:
                            px, py = map(int, point)
                            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

    if detections:
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    heatmap = np.clip(heatmap, 0, 255)
    heatmap_vis = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        motion_heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined_heatmap = cv2.addWeighted(motion_heatmap, 0.5, heatmap_vis, 0.5, 0)
        blended = cv2.addWeighted(frame, 0.6, combined_heatmap, 0.4, 0)
    else:
        blended = frame

    for (ex1, ey1, ex2, ey2) in entry_zones:
        cv2.rectangle(blended, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)

    cv2.putText(blended, "Left-click: Draw | Right-click: Erase | 'q' to quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Video Stream", blended)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
