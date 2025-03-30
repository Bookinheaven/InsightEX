import cv2
import numpy as np
import time
import os
import json
import random
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ---------------------
# Configuration
# ---------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_PATH = os.path.join(CURRENT_DIR, "Data", "ShoppingCCTVVideo.mp4")
ZONES_JSON_FILE = "zones.json"  # Single file to store zones for all sources
OPENVINO_MODEL_DIR = "yolov8m-pose_openvino_model"  # Folder with the exported OpenVINO model
MODEL_NAME = "yolov8m-pose.pt"

DETECTION_CONF_THRESHOLD = 0.5
KEYPOINT_CONF_THRESHOLD = 0.3
ALERT_TIME_SECONDS = 20  # Alert threshold in seconds
# ALERT_TIME_SECONDS = 120

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

# Global variables for drawing and tracking zones
entry_zones = []  # will hold zones for the current source
drawing = False
start_x, start_y = -1, -1

# For color-coding each track ID and tracking times
track_colors = {}
track_start_times = {}  # time when a track was first detected (in video seconds)
track_last_alerts = {}  # time when the last alert was printed for each track

# Global dictionary for ReID feature history per track
track_feature_history = {}

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
# Zones File Handling
# ---------------------
def load_zones_file():
    """Load the zones JSON file that stores zones for all sources."""
    if os.path.exists(ZONES_JSON_FILE) and os.path.getsize(ZONES_JSON_FILE) > 0:
        try:
            with open(ZONES_JSON_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading zones file:", e)
    return {}

def save_zones_file(zones_data):
    """Save the entire zones dictionary to the JSON file."""
    try:
        with open(ZONES_JSON_FILE, "w") as f:
            json.dump(zones_data, f, indent=4)
    except Exception as e:
        print("Error saving zones file:", e)

def get_zones_for_source(source_key):
    zones_data = load_zones_file()
    return zones_data.get(source_key, [])

def update_zones_for_source(source_key, zones):
    zones_data = load_zones_file()
    zones_data[source_key] = zones
    save_zones_file(zones_data)

# ---------------------
# Mouse Callback for Entry Zones
# ---------------------
def draw_entry_zone(event, x, y, flags, param):
    global drawing, start_x, start_y, entry_zones
    source_key = param  # The source key is passed as the callback parameter
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
            update_zones_for_source(source_key, entry_zones)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove any zones that contain the clicked point
        new_zones = []
        for (zx1, zy1, zx2, zy2) in entry_zones:
            if not (zx1 <= x <= zx2 and zy1 <= y <= zy2):
                new_zones.append((zx1, zy1, zx2, zy2))
        entry_zones = new_zones
        update_zones_for_source(source_key, entry_zones)

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

# ---------------------
# ReID Feature Extraction Functions
# ---------------------
def extract_reid_feature(frame, bbox):
    """
    Extracts a feature vector for the given bounding box from the frame.
    This is a placeholder function using a color histogram.
    Replace this with your own ReID model for better performance.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Convert crop to HSV and compute a 3D color histogram
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    feature = hist.flatten()
    return feature

def update_track_features(track_id, new_feature, max_history=5):
    """
    Maintains a history of feature vectors for a track and returns an averaged feature.
    """
    if new_feature is None:
        return None
    if track_id not in track_feature_history:
        track_feature_history[track_id] = []
    track_feature_history[track_id].append(new_feature)
    if len(track_feature_history[track_id]) > max_history:
        track_feature_history[track_id].pop(0)
    avg_feature = np.mean(track_feature_history[track_id], axis=0)
    return avg_feature

# ---------------------
# Main Program
# ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="InsightEX: Video/Cam stream processing with YOLOv8 Pose & DeepSORT."
    )
    parser.add_argument("--video", type=str, help="Path to input video file.")
    parser.add_argument("--camera", type=int, help="Camera index (e.g., 0 for default webcam).")
    args = parser.parse_args()

    # Determine video source and create a unique key for zones storage
    source_key = ""
    video_source = None
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file '{args.video}' does not exist.")
            return
        video_source = args.video
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        source_key = base_name
    elif args.camera is not None:
        video_source = args.camera
        source_key = f"camera{args.camera}"
    else:
        # Fall back to default video path
        if not os.path.exists(DEFAULT_VIDEO_PATH):
            print(f"Error: Default video file '{DEFAULT_VIDEO_PATH}' not found and no --video or --camera provided.")
            return
        video_source = DEFAULT_VIDEO_PATH
        base_name = os.path.splitext(os.path.basename(DEFAULT_VIDEO_PATH))[0]
        source_key = base_name

    print(f"Using zones for source: {source_key}")

    # Load zones for the current source from the shared zones file.
    global entry_zones
    entry_zones = get_zones_for_source(source_key)

    check_Model_Exists()

    model = YOLO(OPENVINO_MODEL_DIR)

    tracker = DeepSort(max_age=200, n_init=3, max_iou_distance=0.7)

    print("Attempting to open video source:", video_source)
    cap = None
    if args.video:
        cap = cv2.VideoCapture(video_source)
    elif args.camera is not None:
        cap = cv2.VideoCapture(video_source)
    else:
        if video_source == DEFAULT_VIDEO_PATH:
            cap = cv2.VideoCapture(video_source)
    if not cap or not cap.isOpened():
        print(f"Error: Cannot open video stream: {video_source}")
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.gect(cv2.CAP_PROP_FRAME_WIDTH))
    heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

    cv2.namedWindow("Insight", cv2.WINDOW_NORMAL)
    # Pass the source_key as the callback parameter so drawing functions update the right zone list.
    cv2.setMouseCallback("Insight", draw_entry_zone, source_key)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed to read frame.")
            break

        # Run inference using YOLOv8 Pose model (with verbose disabled).
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
                if cls_id != 0:  # Skip non-person detections.
                    continue

                conf = float(box.conf[0])
                if conf < DETECTION_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = (x2 - x1), (y2 - y1)

                # Update heatmap only if detection is NOT inside an entry zone.
                if not is_in_entry_zone(x1, y1, x2, y2):
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if 0 <= center_x < frame_w and 0 <= center_y < frame_h:
                        heatmap[center_y, center_x] += 1

                conf_list.append(conf)
                detections.append([[x1, y1, w, h], conf])

                # Unconditionally update heatmap over the detection region.
                heatmap[y1:y2, x1:x2] += 0.5

                # Draw pose on the frame if keypoints are available.
                if len(all_keypoints) > i:
                    keypoints = all_keypoints.data[i].cpu().numpy()
                    draw_pose_keypoints_and_skeleton(frame, keypoints)

        # Update DeepSORT tracker.
        tracks = tracker.update_tracks(detections, frame=frame)

        # Get current video playback time in seconds.
        video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        current_ids = set()
        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                track_id = track.track_id
                current_ids.add(track_id)
                # Set the start time based on video time if not already set.
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

                # --- Enhanced ReID: Extract and update feature history ---
                bbox = [x1, y1, x2, y2]
                reid_feature = extract_reid_feature(frame, bbox)
                avg_feature = update_track_features(track_id, reid_feature)
                # For debugging, you might print the norm of the averaged feature
                if avg_feature is not None:
                    feature_norm = np.linalg.norm(avg_feature)
                    cv2.putText(frame, f"F:{feature_norm:.1f}", (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # ----------------------------------------------------------

                # Alert logic: print alert only once per ALERT_TIME_SECONDS interval.
                if elapsed >= ALERT_TIME_SECONDS:
                    last_alert = track_last_alerts.get(track_id, 0)
                    if (video_time_s - last_alert) >= ALERT_TIME_SECONDS:
                        print(f"ALERT: at ({x1}, {y1}) People need assistance")
                        track_last_alerts[track_id] = video_time_s

        # Remove start times, alert times, and feature histories for tracks no longer present.
        for tid in list(track_start_times.keys()):
            if tid not in current_ids:
                del track_start_times[tid]
                if tid in track_last_alerts:
                    del track_last_alerts[tid]
                if tid in track_feature_history:
                    del track_feature_history[tid]

        # Calculate FPS based on system time.
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        avg_conf = np.mean(conf_list) if conf_list else 0

        # Zero out heatmap inside entry zones.
        for (zx1, zy1, zx2, zy2) in entry_zones:
            zx1 = max(0, zx1)
            zy1 = max(0, zy1)
            zx2 = min(frame_w, zx2)
            zy2 = min(frame_h, zy2)
            heatmap[zy1:zy2, zx1:zx2] = 0

        # Prepare heatmap overlay.
        heatmap = np.clip(heatmap, 0, 255)
        heatmap_vis = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.8, heatmap_vis, 0.2, 0)

        # Draw entry zones on the overlay.
        for (zx1, zy1, zx2, zy2) in entry_zones:
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)

        # Display metrics.
        cv2.putText(overlay, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Inference: {inference_time:.1f} ms", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Avg Conf: {avg_conf:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, "Left-click: Draw zone | Right-click: Erase zone | 'q' to quit",
                    (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Insight", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
