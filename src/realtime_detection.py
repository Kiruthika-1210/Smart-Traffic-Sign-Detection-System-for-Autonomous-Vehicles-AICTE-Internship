import os
import cv2
import time
import argparse
from datetime import datetime
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Realtime traffic sign detection (webcam/video) using YOLOv8")
    p.add_argument("--model", type=str, default="../models/traffic_sign_model/weights/best.pt",
                   help="path to weights (default: ../models/traffic_sign_model/weights/best.pt)")
    p.add_argument("--source", type=str, default="0",
                   help="0 or webcam index for webcam, or path to video/image folder. (default: 0)")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--save", action="store_true", help="save output video to results/")
    p.add_argument("--show", action="store_true", help="show window (default False on headless systems)")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    return p.parse_args()

def get_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def main():
    args = parse_args()

    # Resolve source
    source = args.source
    use_webcam = False
    try:
        src_int = int(source)
        use_webcam = True
        source = src_int
    except Exception:
        use_webcam = False

    model_path = os.path.expanduser(args.model)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    print(f"✅ Loading model: {model_path}")
    model = YOLO(model_path)  # loads the model
    model.fuse()  # fuse conv+bias where possible to speed up inference

    # open cv capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    # fps calculation
    prev_time = 0.0

    # prepare output folder if saving
    writer = None
    out_dir = None
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("results", f"realtime_{ts}")
        os.makedirs(out_dir, exist_ok=True)

    # get video info (width/height/fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    if args.save:
        out_path = os.path.join(out_dir, "output.mp4")
        print(f"🔁 Saving output video to: {out_path}")
        writer = get_video_writer(out_path, fps if fps>0 else 20.0, width, height)

    names = model.names  # class id -> name

    print("▶ Running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❗ End of stream or cannot fetch frame.")
            break

        start = time.time()

        # run inference on the frame (model accepts numpy array)
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)  # returns list of Results (we pass single frame)
        # results can contain one Results object (for the input frame)
        if len(results) > 0:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # box.xyxy is tensor-like; convert to list
                    try:
                        xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # [x1, y1, x2, y2]
                    except Exception:
                        xyxy = box.xyxy.numpy().astype(int)[0]
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
                    cls = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
                    label = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]

                    # draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (12, 210, 12), 2)
                    txt = f"{label} {conf:.2f}"
                    # text background
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (12, 210, 12), -1)
                    cv2.putText(frame, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # FPS
        end = time.time()
        fps_val = 1.0 / (end - prev_time) if prev_time else 0.0
        prev_time = end
        fps_text = f"FPS: {fps_val:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # show
        if args.show:
            cv2.imshow("Realtime Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # write
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"✅ Video saved: {out_path}")
        # also save a single example frame for README
        example_path = os.path.join(out_dir, "example_frame.jpg")
        cv2.imwrite(example_path, frame)
        print(f"✅ Example frame saved: {example_path}")

    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    main()
