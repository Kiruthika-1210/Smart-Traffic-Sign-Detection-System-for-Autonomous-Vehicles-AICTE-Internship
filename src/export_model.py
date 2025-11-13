from ultralytics import YOLO
import os

def export_model():
    model_path = "../models/traffic_sign_model/weights/best.pt"

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    print(f"✅ Loading model from: {model_path}")
    model = YOLO(model_path)

    print("🔁 Exporting to ONNX…")
    model.export(format="onnx")

    print("🔁 Exporting to TorchScript…")
    model.export(format="torchscript")

    print("🔁 Exporting to OpenVINO (optional)…")
    model.export(format="openvino")

    print("🎉 Export completed! Check the folder:")
    print("➡ models/traffic_sign_model/")

if __name__ == "__main__":
    export_model()
