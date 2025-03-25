from ultralytics import YOLO

# Load the YOLOv8 model (change 'yolov8s.pt' to 'yolov8m.pt' or 'yolov8l.pt' if needed)
model = YOLO("yolov8m.pt")

# Train the model
model.train(
    data="dataset.yaml",  # Path to dataset YAML file
    epochs=100,                  # Number of epochs
    imgsz=640,                   # Image size
    batch=16,                     # Batch size (adjust based on GPU memory)
    workers=4,                    # Number of CPU workers
    device="cuda"                 # Use GPU (change to "cpu" if no GPU)
)

# Save the trained model
model.export(format="onnx")  # Optional: Export to ONNX format
