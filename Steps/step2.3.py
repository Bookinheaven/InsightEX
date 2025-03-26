# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from openvino import Core
#
# # Load YOLOv8 model with OpenVINO optimization
# ie = Core()
# model = YOLO("yolov8m.pt")  # Load OpenVINO YOLO model
#
# # Initialize DeepSORT tracker
# tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)
#
# # Open video file
# cap = cv2.VideoCapture("../Data/ShoppingCCTVVideo.mp4")
#
# # Initialize heatmap storage
# heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)
#
# # Entry zones (to be drawn by the user)
# entry_zones = []
# drawing = False
# start_x, start_y = -1, -1
#
# # Mouse callback function
# def draw_entry_zone(event, x, y, flags, param):
#     global start_x, start_y, drawing, entry_zones
#
#     if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
#         start_x, start_y = x, y
#         drawing = True
#
#     elif event == cv2.EVENT_LBUTTONUP:  # Finish drawing
#         entry_zones.append((start_x, start_y, x, y))
#         drawing = False
#
# # Function to check if a bounding box is in an entry zone
# def is_in_entry_zone(x1, y1, x2, y2):
#     for (ex1, ey1, ex2, ey2) in entry_zones:
#         if x1 >= ex1 and x2 <= ex2 and y1 >= ey1 and y2 <= ey2:
#             return True
#     return False
#
# # Set up mouse event for selecting entry zones
# cv2.namedWindow("Draw Entry Zones")
# cv2.setMouseCallback("Draw Entry Zones", draw_entry_zone)
#
# # Wait for user to draw entry zones
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Show drawn zones
#     for (ex1, ey1, ex2, ey2) in entry_zones:
#         cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)
#
#     cv2.putText(frame, "Draw entry zones with the mouse, press 's' to start tracking",
#                 (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     cv2.imshow("Draw Entry Zones", frame)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("s"):  # Press 's' to start tracking
#         break
#     elif key == ord("q"):  # Press 'q' to quit
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
#
# cv2.destroyWindow("Draw Entry Zones")
#
# # Reset video capture
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# prev_gray = None
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to grayscale for motion detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Run YOLO object detection
#     results = model(frame)
#
#     # Extract bounding boxes for persons (class 0)
#     detections = []
#     for result in results:
#         for box in result.boxes:
#             if int(box.cls) == 0:  # Class 0 = Person
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 w, h = x2 - x1, y2 - y1
#                 score = float(box.conf[0])
#
#                 # Ignore persons in entry zones
#                 if score > 0.4 and not is_in_entry_zone(x1, y1, x2, y2):
#                     detections.append([[x1, y1, w, h], score])
#                     heatmap[y1:y2, x1:x2] += 1  # Update heatmap
#
#     # Normalize heatmap
#     heatmap = np.clip(heatmap, 0, 255)
#     heatmap_vis = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
#
#     # Motion Detection using Optical Flow
#     if prev_gray is not None:
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         hsv = np.zeros_like(frame)
#         hsv[..., 1] = 255
#
#         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         hsv[..., 0] = ang * 180 / np.pi / 2
#         hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#
#         motion_heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#         # Combine motion heatmap & crowd heatmap
#         combined_heatmap = cv2.addWeighted(motion_heatmap, 0.5, heatmap_vis, 0.5, 0)
#
#         # Overlay with original frame
#         blended = cv2.addWeighted(frame, 0.6, combined_heatmap, 0.4, 0)
#
#         # Draw entry zones
#         for (ex1, ey1, ex2, ey2) in entry_zones:
#             cv2.rectangle(blended, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
#
#         cv2.imshow("Crowd & Motion Heatmap", blended)
#
#     prev_gray = gray  # Update previous frame
#
#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO

# Load YOLOv8 PyTorch model
model = YOLO("yolov8m-pose.pt", task="pose")

# Export to OpenVINO format
model.export(format="openvino")

# Load the exported OpenVINO model
ov_model = YOLO("yolov8m_openvino_model", task="pose")
