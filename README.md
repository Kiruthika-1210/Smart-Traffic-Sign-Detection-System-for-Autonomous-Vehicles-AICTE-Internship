Smart-Traffic-Sign-Detection-System-for-Autonomous-Vehicles-AICTE-Internship
A deep learning–based Traffic Sign Detection System built using YOLOv8 and OpenCV. This project enables real-time detection and classification of traffic signs to assist autonomous vehicles in understanding and responding to road conditions.
WEEK1
What This Project Intends to Do ?
-> Detect and classify multiple traffic signs from images or video streams.
-> Help autonomous vehicles interpret road conditions accurately.
-> Provide real-time feedback for safe navigation.
-> Demonstrate the application of AI in computer vision and intelligent transport systems.


WEEK 2 – Dataset Preparation, Model Training & Performance Analysis
During Week 2, the focus shifted from conceptual understanding to hands-on implementation of the Traffic Sign Detection System. The major objectives were to prepare the dataset, train the YOLOv8 detection model, test its performance, and generate analytical visualizations.

1. Dataset Preparation
This week involved converting the raw dataset into a structure compatible with YOLOv8.
Key Tasks Completed
Organized Dataset
Created separate folders for images and annotations.
Cleaned unwanted files and verified dataset integrity.
Train/Test Split
Automatically split the dataset into train and test sets using split_dataset.py.
Ensured correct movement of images and corresponding XML annotations.
Annotation Conversion
Converted Pascal VOC XML files into YOLO format using convert_xml_to_yolo.py.
Extracted class labels:
crosswalk, speedlimit, stop, trafficlight.
Created Dataset YAML File
Configured data.yaml with paths to images and class names for YOLO training.

2. Model Training (YOLOv8)
Trained the traffic sign detection model using YOLOv8n for efficiency.
Training Details
Model Used: YOLOv8n (Nano variant)
Training Device: CPU
Epochs: 30
Batch Size: 16
Image Size: 640×640
Results
Successfully trained the model across 30 epochs.
Achieved stable improvement across:
box loss
classification loss
DFL loss
precision
recall
mAP50
mAP50-95
Best weights saved at:
models/traffic_sign_model/weights/best.pt

3. Model Testing
Used the trained model for inference on both:
Single images
Folder of images
Sample inference successfully detected a traffic light from a test image.
Output is saved in the results/ directory.

4. Performance Visualization
Generated detailed analytical plots using plot_metrics.py:
Graphs Produced
Training Loss Curves
Validation Loss Curves
Precision vs. Recall
mAP50 & mAP50-95 over epochs
These graphs provide a clear view of model convergence and performance improvements.
Saved at:
results/training_losses.png
results/validation_losses.png
results/precision_recall.png
results/map_scores.png
results/training_losses.png
results/validation_losses.png
results/precision_recall.png
results/map_scores.png
