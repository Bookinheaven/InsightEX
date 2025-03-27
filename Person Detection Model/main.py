import cv2
import numpy as np
import json
import os
import time
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from openvino.runtime import Core
from deepface import DeepFace

# Configuration
MODEL_NAME = "yolov8m-pose.pt"
OPENVINO_MODEL_DIR = "yolov8m-pose_openvino_model"
VIDEO_PATH = "../Data/ShoppingCCTVVideo.mp4"
ZONES_FILE = "../Steps/zones.json"
STAFF_FACES_DIR = "staff_faces/"
ALERT_TIME = 120

drawing = False
start_x, start_y = -1, -1


def initialize_models():
    current_dir = os.getcwd()  # Get current working directory
    model_path = os.path.abspath(MODEL_NAME)  # Absolute path for model
    openvino_model_path = os.path.join(os.path.abspath(OPENVINO_MODEL_DIR))

    # Ensure model exists
    if not os.path.exists(model_path):
        print(f"{MODEL_NAME} not found in {current_dir}. Downloading...")
        model = YOLO(MODEL_NAME)
        model.download()

        # Wait until the model is completely downloaded
        while not os.path.exists(model_path):
            print(f"Waiting for {MODEL_NAME} download to complete...")
            time.sleep(2)

    # Ensure OpenVINO model exists
    if not os.path.exists(openvino_model_path):
        print("Exporting model to OpenVINO format...")
        model = YOLO(model_path, task="pose")
        model.export(format="openvino")

        # Wait until export is complete
        while not os.path.exists(openvino_model_path):
            print("Waiting for OpenVINO model export to complete...")
            time.sleep(2)
    else:
        print("OpenVINO model already exists.")

    ie = Core()
    available_devices = ie.available_devices
    device_priority = ["NPU", "VPU", "HDDL", "MYRIAD", "GPU", "GNA", "CPU"]
    device = next((d for d in device_priority if d in available_devices), "CPU")
    print(f"Using OpenVINO device: {device}")

    # Load OpenVINO Model
    ov_model = YOLO(openvino_model_path)  # Removed task="pose" to prevent format errors
    tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

    return ov_model, tracker


# Load and manage entry zones
def load_zones():
    if os.path.exists(ZONES_FILE) and os.path.getsize(ZONES_FILE) > 0:
        try:
            with open(ZONES_FILE, "r") as f:
                zones = json.load(f)
                if isinstance(zones, list) and all(isinstance(zone, list) and len(zone) == 4 for zone in zones):
                    return zones  # ✅ Correct format
                else:
                    print("⚠️ Invalid zone format in JSON. Resetting zones.")
                    return []  # 🛑 Reset if invalid
        except json.JSONDecodeError:
            print("⚠️ JSON Decode Error! Resetting zones.")
            return []
    return []


def save_zones(zones):
    with open(ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=4)


# Mouse callback function to draw entry zones
def draw_entry_zone(event, x, y, flags, param):
    global drawing, start_x, start_y, entry_zones

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pass  # Allows dynamic drawing (optional visualization step)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        if start_x != end_x and start_y != end_y:
            entry_zones.append([min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)])
            save_zones(entry_zones)
    elif event == cv2.EVENT_RBUTTONDOWN:
        entry_zones = [zone for zone in entry_zones if not (zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3])]
        save_zones(entry_zones)


def is_inside_entry_zone(cx, cy, entry_zones):
    """Check if a point (cx, cy) is inside any of the entry zones."""
    for zx1, zy1, zx2, zy2 in entry_zones:
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return True  # Point is inside an entry zone
    return False


def main():
    global entry_zones
    model, tracker = initialize_models()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    entry_zones = load_zones()
    people_in_zones = defaultdict(lambda: deque(maxlen=int(ALERT_TIME * fps)))

    cv2.namedWindow("Video Stream")
    cv2.setMouseCallback("Video Stream", draw_entry_zone)

    # Initialize heatmap with correct shape
    heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

    tracking_stats = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply decay to heatmap to avoid saturation
        heatmap *= 0.95  # Adjust decay factor as needed

        # Draw entry zones on the frame
        for zone in entry_zones:
            if len(zone) == 4:
                zx1, zy1, zx2, zy2 = zone
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 0), 2)

        # Run YOLOv8 Pose Model
        results = model(frame, verbose=False)[0]

        detections = []
        for detection, keypoints in zip(results.boxes.data.tolist(), results.keypoints.data.cpu().numpy()):
            x1, y1, x2, y2, conf, class_id = detection[:6]
            if conf > 0.5 and int(class_id) == 0:  # Person detection
                bbox = (int(x1), int(y1), int(x2), int(y2))
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf])

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Track detected persons
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)

                # Draw Tracking ID
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Compute person's center
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Ensure (cx, cy) is within valid image dimensions
                cx = max(0, min(cx, frame_width - 1))
                cy = max(0, min(cy, frame_height - 1))

                # Update heatmap if the person is NOT inside the entry zone
                if not is_inside_entry_zone(cx, cy, entry_zones):
                    heatmap[cy, cx] = min(heatmap[cy, cx] + 1, 255)

        # Normalize and display heatmap
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_norm), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        cv2.imshow("Video Stream", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
