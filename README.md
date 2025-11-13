TITLE: Smart Traffic Sign Detection System for Autonomous Vehicles (AICTE Internship)
Tech Stack: YOLOv8, OpenCV, Python, Real-time Detection

INTRODUCTION:
This project implements a complete Traffic Sign Detection System using YOLOv8. The goal is to detect and classify multiple road signs in real time for autonomous driving applications. The system can process images, video files, and live webcam feeds. It demonstrates how computer vision and deep learning can assist self-driving cars in interpreting road environments.

PROJECT FEATURES:
Dataset preprocessing, annotation conversion, and YOLO-compatible formatting.
Model training using YOLOv8 on a large, labeled traffic sign dataset.
Performance evaluation using precision, recall, mAP50, and mAP50-95 metrics.
Real-time detection using a webcam or video input with bounding boxes, labels, confidence scores, and FPS displayed.
Visualized metrics: loss curves, mAP curves, precision-recall graphs.

FOLDER STRUCTURE:
src/ → all scripts (training, testing, conversion, real-time detection)
data/ → dataset (images, labels, annotation XML before conversion)
models/ → trained YOLOv8 weights (best.pt and last.pt)
results/ → detection outputs, graphs, and evaluation metrics
notebooks/ → optional Jupyter notebooks for experiments

HOW TO SET UP:
Install dependencies using: pip install -r requirements.txt
Place dataset inside the data folder.
Convert XML annotations into YOLO format using convert_xml_to_yolo.py
Split the dataset into train and test using split_dataset.py

TRAINING THE MODEL:
Use the training script to train YOLOv8:
python train_model.py
This will generate best.pt under models/traffic_sign_model/weights/.
The training automatically logs metrics and saves graphs.

TESTING THE MODEL ON IMAGES:
Use detect_image.py to test any single image or folder of images.
The script saves predictions to results/.

REAL-TIME DETECTION:
The realtime_detection.py script allows detection from:
– webcam
– video file
– image sequence
It draws bounding boxes, labels, confidence scores, and FPS.
Example usage:
python realtime_detection.py --source 0 --show
To save the output as a video:
python realtime_detection.py --source your_video.mp4 --save --show

MODEL PERFORMANCE (SUMMARY):
Precision: high (especially speedlimit and stop signs)
Recall: high for common signs, slightly lower for rare categories
mAP50: excellent
mAP50-95: strong performance overall
Overall accuracy is suitable for real-time autonomous navigation prototypes.

RESULT SAMPLES:
Include saved detections from results/ folder.
Include example_frame.jpg from real-time detection.

APPLICATIONS:
Autonomous vehicles
Driver assistance systems
Smart road monitoring
Traffic rule enforcement automation

LIMITATIONS:
Certain rare signs (U-turn, pedestrian crossing) require more training data.
Low-light or blurred frames can reduce accuracy.
Adding more images can further improve model robustness.

FUTURE IMPROVEMENTS:
Add more diverse training samples to improve rare-class accuracy.
Optimize the model using ONNX or TensorRT for faster deployment.
Deploy the model in a web dashboard or mobile app.
Implement a full driver-assistance system pipeline.

CONCLUSION:
This project successfully completes the AICTE Traffic Sign Detection internship by building a professional-grade detection system from scratch. The system is fully functional, real-time capable, and ready for further deployment or research expansions.
