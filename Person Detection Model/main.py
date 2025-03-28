import cv2
import numpy as np
import time
import os
import json
import random
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO  # Using ultralytics for inference as in your example

# ---------------------
# Configuration
# ---------------------
VIDEO_PATH = "../Data/ShoppingCCTVVideo.mp4"
OPENVINO_MODEL_DIR = "yolov8m-pose_openvino_model"  # Folder with the exported OpenVINO model
ZONES_FILE = "zones.json"

DETECTION_CONF_THRESHOLD = 0.5
KEYPOINT_CONF_THRESHOLD = 0.3
ALERT_TIME_SECONDS = 120  # 2 minutes

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
# For tracking the first appearance of each track (in video seconds)
track_start_times = {}

def get_track_color(track_id):
    """Assign a random color to each unique track ID."""
    if track_id not in track_colors:
        track_colors[track_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return track_colors[track_id]

# ---------------------
# Utility Functions for Entry Zones
# ---------------------
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

# ---------------------
# Pose Drawing Function
# ---------------------
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

# ---------------------
# Main Program
# ---------------------
def main():
    global entry_zones
    entry_zones = load_zones()

    # 1. Initialize the YOLOv8 Pose model (exported to OpenVINO) using ultralytics
    model = YOLO(OPENVINO_MODEL_DIR)

    # 2. Initialize DeepSORT for tracking with adjusted parameters for better track stability
    # Increase max_age so that tracks persist longer even if detections are momentarily lost.
    tracker = DeepSort(max_age=100, n_init=3, max_iou_distance=0.7)

    # 3. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Video Stream", draw_entry_zone)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference using YOLOv8 Pose model (with verbose disabled)
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_time = (time.time() - inference_start) * 1000  # ms

        detections = []
        conf_list = []
        for result in results:
            if not hasattr(result, 'boxes'):
                continue

            boxes = result.boxes
            all_keypoints = result.keypoints  # shape: [num_persons, 17, 3]
            if all_keypoints is None:
                all_keypoints = []

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id != 0:  # Skip non-person detections
                    continue

                conf = float(box.conf[0])
                if conf < DETECTION_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = (x2 - x1), (y2 - y1)

                # Update heatmap only if detection is NOT inside an entry zone
                if not is_in_entry_zone(x1, y1, x2, y2):
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if 0 <= center_x < frame_w and 0 <= center_y < frame_h:
                        heatmap[center_y, center_x] += 1

                conf_list.append(conf)
                detections.append([[x1, y1, w, h], conf])

                # Unconditionally update heatmap over the detection region
                heatmap[y1:y2, x1:x2] += 0.5

                # Draw pose on the frame if keypoints are available
                if len(all_keypoints) > i:
                    keypoints = all_keypoints.data[i].cpu().numpy()
                    draw_pose_keypoints_and_skeleton(frame, keypoints)

        # Update DeepSORT tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Get current video playback time in seconds (video-based time)
        video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        current_ids = set()
        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                track_id = track.track_id
                current_ids.add(track_id)
                # Set the start time based on video time if not already set
                if track_id not in track_start_times:
                    track_start_times[track_id] = video_time_s
                elapsed = video_time_s - track_start_times[track_id]

                x1, y1, x2, y2 = map(int, track.to_ltrb())
                color = get_track_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cx = (x1 + x2) // 2
                text_y = max(0, y1 - 10)
                cv2.putText(frame, f"ID {track_id} Time: {int(elapsed)}s", (cx - 20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # If elapsed >= ALERT_TIME_SECONDS, draw an alert
                if elapsed >= ALERT_TIME_SECONDS:
                    print(f"ALERT: at ({x1}, {y1}) People need assistance")

        # Remove start times for tracks no longer present
        for tid in list(track_start_times.keys()):
            if tid not in current_ids:
                del track_start_times[tid]

        # Calculate FPS based on system time (this may be lower if processing is slow)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        avg_conf = np.mean(conf_list) if conf_list else 0

        # Zero out heatmap inside entry zones
        for (zx1, zy1, zx2, zy2) in entry_zones:
            zx1 = max(0, zx1)
            zy1 = max(0, zy1)
            zx2 = min(frame_w, zx2)
            zy2 = min(frame_h, zy2)
            heatmap[zy1:zy2, zx1:zx2] = 0

        # Prepare heatmap overlay exactly as specified:
        heatmap = np.clip(heatmap, 0, 255)
        heatmap_vis = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.8, heatmap_vis, 0.2, 0)

        # Draw entry zones on the overlay
        for (zx1, zy1, zx2, zy2) in entry_zones:
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)

        # Display metrics on the overlay
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
