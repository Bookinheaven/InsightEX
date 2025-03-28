import cv2
import numpy as np
import time
import os
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from openvino.runtime import Core
import random

VIDEO_PATH = "../Data/ShoppingCCTVVideo.mp4"
OPENVINO_MODEL_DIR = "yolov8m-pose_openvino_model"  # Folder containing the exported OpenVINO model
ZONES_FILE = "zones.json"
MODEL_NAME = "yolov8m-pose.pt"
OPENVINO_MODEL_DIR = "yolov8m-pose_openvino_model"

DETECTION_CONF_THRESHOLD = 0.5
KEYPOINT_CONF_THRESHOLD = 0.3

SKELETON_CONNECTIONS = [
    (0, 1),  # nose -> left eye
    (0, 2),  # nose -> right eye
    (1, 3),  # left eye -> left ear
    (2, 4),  # right eye -> right ear
    (5, 6),  # left shoulder -> right shoulder
    (5, 7),  # left shoulder -> left elbow
    (7, 9),  # left elbow -> left wrist
    (6, 8),  # right shoulder -> right elbow
    (8, 10), # right elbow -> right wrist
    (5, 11), # left shoulder -> left hip
    (6, 12), # right shoulder -> right hip
    (11, 12),# left hip -> right hip
    (11, 13),# left hip -> left knee
    (13, 15),# left knee -> left ankle
    (12, 14),# right hip -> right knee
    (14, 16) # right knee -> right ankle
]

entry_zones = []
drawing = False
start_x, start_y = -1, -1

# For color-coding each track ID
track_colors = {}

def get_track_color(track_id):
    """Assign a random color to each unique track ID."""
    if track_id not in track_colors:
        track_colors[track_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return track_colors[track_id]

def load_zones():
    if os.path.exists(ZONES_FILE) and os.path.getsize(ZONES_FILE) > 0:
        try:
            with open(ZONES_FILE, "r") as f:
                zones = json.load(f)
                return [tuple(zone) for zone in zones]  # convert lists to tuples
        except Exception as e:
            print("Error loading zones:", e)
    return []

def save_zones(zones):
    try:
        with open(ZONES_FILE, "w") as f:
            json.dump(zones, f, indent=4)
    except Exception as e:
        print("Error saving zones:", e)

def draw_entry_zone(event, x, y, flags, param):
    global drawing, start_x, start_y, entry_zones
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        if start_x != end_x and start_y != end_y:
            zone = (min(start_x, end_x), min(start_y, end_y),
                    max(start_x, end_x), max(start_y, end_y))
            entry_zones.append(zone)
            save_zones(entry_zones)
    elif event == cv2.EVENT_RBUTTONDOWN:
        new_zones = []
        for (zx1, zy1, zx2, zy2) in entry_zones:
            if not (zx1 <= x <= zx2 and zy1 <= y <= zy2):
                new_zones.append((zx1, zy1, zx2, zy2))
        entry_zones = new_zones
        save_zones(entry_zones)

def is_in_entry_zone(x1, y1, x2, y2):
    """Return True if the bounding box is entirely within any entry zone."""
    for (zx1, zy1, zx2, zy2) in entry_zones:
        if x1 >= zx1 and x2 <= zx2 and y1 >= zy1 and y2 <= zy2:
            return True
    return False

def draw_pose_keypoints_and_skeleton(frame, keypoints):
    valid_points = [None] * len(keypoints)
    for idx, (kx, ky, kconf) in enumerate(keypoints):
        if (kconf > KEYPOINT_CONF_THRESHOLD and
            not (int(kx) == 0 and int(ky) == 0) and
            0 <= int(kx) < frame.shape[1] and
            0 <= int(ky) < frame.shape[0]):
            cv2.circle(frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)
            valid_points[idx] = (int(kx), int(ky))

    for i, j in SKELETON_CONNECTIONS:
        if i < len(valid_points) and j < len(valid_points):
            pt1 = valid_points[i]
            pt2 = valid_points[j]
            if pt1 is not None and pt2 is not None:
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

def check_Model_Exists():
    # Download PyTorch model if not present.
    if not os.path.exists(MODEL_NAME):
        print(f"{MODEL_NAME} not found. Downloading...")
        model = YOLO(MODEL_NAME)
        while not os.path.exists(MODEL_NAME):
            print("Waiting for model download...")
            time.sleep(2)
    else:
        print(f"{MODEL_NAME} already exists.")
    # Export to OpenVINO if the directory is not present.
    if not os.path.exists(OPENVINO_MODEL_DIR):
        print("OpenVINO model not found. Exporting...")
        model = YOLO(MODEL_NAME, task="pose")
        model.export(format="openvino")
        while not os.path.exists(OPENVINO_MODEL_DIR):
            print("Waiting for OpenVINO export...")
            time.sleep(2)
        print("Export completed.")
    else:
        print("OpenVINO model already exists.")



def main():
    check_Model_Exists()
    global entry_zones
    entry_zones = load_zones()

    # 1. Initialize OpenVINO runtime and pick a device
    ie = Core()
    available_devices = ie.available_devices
    device_priority = ["GPU", "HDDL", "MYRIAD", "CPU"]
    device = next((d for d in device_priority if d in available_devices), "CPU")
    print("Using OpenVINO device:", device)

    # 2. Load the OpenVINO-exported YOLOv8 Pose Model
    #    Ensure 'OPENVINO_MODEL_DIR' is the folder with the .xml/.bin files
    model = YOLO(OPENVINO_MODEL_DIR)

    # 3. Initialize DeepSORT
    tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

    # 4. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")

    # 5. Create Heatmap
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

    # Resizable window
    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Video Stream", draw_entry_zone)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Blur the heatmap for smoother appearance
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), 1)

        # Run inference with verbose=False
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_time = (time.time() - inference_start) * 1000  # ms

        detections = []
        conf_list = []
        for result in results:
            if not hasattr(result, 'boxes'):
                continue

            boxes = result.boxes
            all_keypoints = result.keypoints  # shape [num_persons, 17, 3]
            if all_keypoints is None:
                all_keypoints = []

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id != 0:  # skip non-person
                    continue

                conf = float(box.conf[0])
                if conf < DETECTION_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = (x2 - x1), (y2 - y1)

                # Skip if bounding box is in the entry zone
                if is_in_entry_zone(x1, y1, x2, y2):
                    continue

                # Valid detection for tracking
                conf_list.append(conf)
                detections.append([[x1, y1, w, h], conf])

                # Update heatmap
                heatmap[y1:y2, x1:x2] += 0.5

                # Draw Pose
                if len(all_keypoints) > i:
                    keypoints = all_keypoints.data[i].cpu().numpy()
                    draw_pose_keypoints_and_skeleton(frame, keypoints)

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                # Draw color-coded bounding box
                color = get_track_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Place ID text near the top center
                cx = (x1 + x2) // 2
                text_y = max(0, y1 - 10)
                cv2.putText(frame, f"ID {track_id}", (cx - 20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # Rough "accuracy" metric
        avg_conf = np.mean(conf_list) if conf_list else 0

        # Prepare heatmap overlay
        heatmap = np.clip(heatmap, 0, 255)
        heatmap_vis = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.8, heatmap_vis, 0.2, 0)

        # Draw entry zones
        for (zx1, zy1, zx2, zy2) in entry_zones:
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)

        # Display metrics
        cv2.putText(overlay, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Inference: {inference_time:.1f} ms", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Avg Conf: {avg_conf:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, "Left-click: Draw zone | Right-click: Erase zone | 'q' to quit",
                    (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Video Stream", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
