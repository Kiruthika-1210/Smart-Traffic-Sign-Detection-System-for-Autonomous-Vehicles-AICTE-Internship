import os
import sys
from ultralytics import YOLO
from datetime import datetime

# ------------------------------------------------------------------------------------
# CONFIG: Update ONLY if your model path changes
MODEL_PATH = "../models/traffic_sign_model/weights/best.pt"
OUTPUT_DIR = "../results"
# ------------------------------------------------------------------------------------

def load_model():
    """Load YOLOv8 model from disc."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit()
    print(f"✅ Loaded model: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def detect_on_image(model, image_path):
    """Run detection on a single image."""
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found: {image_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(OUTPUT_DIR, f"image_{timestamp}")

    print(f"🔍 Detecting signs in image: {image_path}")
    results = model.predict(
        source=image_path,
        save=True,
        project=save_dir,
        name="",
        show=False
    )

    print(f"✅ Results saved at: {save_dir}")
    return results


def detect_on_folder(model, folder_path):
    """Run detection on all images in a folder."""
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder not found: {folder_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(OUTPUT_DIR, f"folder_{timestamp}")

    print(f"🔍 Running detection on folder: {folder_path}")
    results = model.predict(
        source=folder_path,
        save=True,
        project=save_dir,
        name="",
        show=False
    )

    print(f"✅ Folder results saved at: {save_dir}")
    return results


def main():
    """Main program entry point."""
    model = load_model()

    print("\nChoose an option:")
    print("1. Detect on a single image")
    print("2. Detect on a folder of images")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        image_path = input("Enter full path to image: ").strip()
        detect_on_image(model, image_path)

    elif choice == "2":
        folder_path = input("Enter full path to folder: ").strip()
        detect_on_folder(model, folder_path)

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
