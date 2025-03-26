import cv2
import numpy as np
import json
import os
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from openvino.runtime import Core

ie = Core()
device = "GPU" if "GPU" in ie.available_devices else "CPU"

model = YOLO("yolov8m-pose_openvino_model/")
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)
cap = cv2.VideoCapture("../Data/ShoppingCCTVVideo.mp4")
heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)

# Entry Zones Storage
entry_zones = []
ZONES_FILE = "zones.json"
drawing = False
start_x, start_y = -1, -1

# People tracking
people_tracking = defaultdict(lambda: {"last_seen": deque(maxlen=120), "is_staff": False, "count": 0})

# Load stored entry zones
def load_zones():
    global entry_zones
    if os.path.exists(ZONES_FILE) and os.path.getsize(ZONES_FILE) > 0:
        with open(ZONES_FILE, "r") as f:
            try:
                entry_zones = json.load(f)
            except json.JSONDecodeError:
                entry_zones = []

def save_zones():
    with open(ZONES_FILE, "w") as f:
        json.dump(entry_zones, f)

# Handle zone drawing and deletion
def draw_entry_zone(event, x, y, flags, param):
    global start_x, start_y, drawing, entry_zones
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        entry_zones.append((start_x, start_y, x, y))
        save_zones()
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        entry_zones = [(ex1, ey1, ex2, ey2) for (ex1, ey1, ex2, ey2) in entry_zones if not (ex1 <= x <= ex2 and ey1 <= y <= ey2)]
        save_zones()

# Check if a person is in an entry zone
def is_in_entry_zone(x1, y1, x2, y2):
    for (ex1, ey1, ex2, ey2) in entry_zones:
        if x1 >= ex1 and x2 <= ex2 and y1 >= ey1 and y2 <= ey2:
            return True
    return False

# Staff recognition placeholder
def is_staff(frame, bbox):
    return False  # To be implemented

cv2.namedWindow("Video Stream")
cv2.setMouseCallback("Video Stream", draw_entry_zone)
prev_gray = None
load_zones()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = model(frame)[0]
    detections = []
    current_people = []
    current_staff = []
    person_id_counter = 0

    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, objectness, class_id = detection[:6]
        if objectness > 0.5 and int(class_id) == 0:
            person_id_counter += 1
            current_people.append((person_id_counter, int(x1), int(y1), int(x2), int(y2)))

            is_staff_member = is_staff(frame, (int(x1), int(y1), int(x2), int(y2)))
            if is_staff_member:
                current_staff.append((person_id_counter, int(x1), int(y1), int(x2), int(y2)))

            people_tracking[person_id_counter]['is_staff'] = is_staff_member
            if not is_in_entry_zone(int(x1), int(y1), int(x2), int(y2)) and not is_staff_member:
                people_tracking[person_id_counter]['count'] += 1
                people_tracking[person_id_counter]['last_seen'].append((int(x1), int(y1), int(x2), int(y2)))
            else:
                people_tracking[person_id_counter]['count'] = 0

            if people_tracking[person_id_counter]['count'] > 120:
                print(f"ALERT: Person {person_id_counter} has been in the same spot for 2 minutes!")

            detections.append([[int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)], objectness])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

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

    prev_gray = gray.copy()
    for (ex1, ey1, ex2, ey2) in entry_zones:
        cv2.rectangle(blended, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)

    cv2.putText(blended, "Left-click: Draw | Right-click: Erase | 'q' to quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Video Stream", blended)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
