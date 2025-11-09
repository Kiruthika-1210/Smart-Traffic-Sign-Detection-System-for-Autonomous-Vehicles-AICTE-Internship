from ultralytics import YOLO
import os

# Path to YAML config
data_yaml = "../data/data.yaml"

# Output directory
output_dir = "../models"
os.makedirs(output_dir, exist_ok=True)

def train_model():
    print("✅ Starting YOLOv8 training...")

    model = YOLO("yolov8n.pt")  # nano model (fastest for internship hardware)

    results = model.train(
        data=data_yaml,
        epochs=30,          # You can increase to 50 for better accuracy
        imgsz=640,
        batch=16,
        name="traffic_sign_model",
        project=output_dir,
        verbose=True
    )

    print("✅ Training completed!")
    print("✅ Best weights saved at: models/traffic_sign_model/weights/best.pt")

if __name__ == "__main__":
    train_model()
