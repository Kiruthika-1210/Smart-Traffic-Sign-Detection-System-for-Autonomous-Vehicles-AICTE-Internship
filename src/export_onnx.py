from ultralytics import YOLO
import os

def export_to_onnx():
    model_path = "../models/traffic_sign_model/weights/best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    print(f"🔄 Loading model: {model_path}")
    model = YOLO(model_path)

    print("📦 Exporting model to ONNX...")
    model.export(format="onnx")

    print("✅ Export complete!")
    print("📁 Check folder: models/traffic_sign_model/weights/")
    print("You will see: best.onnx")

if __name__ == "__main__":
    export_to_onnx()
