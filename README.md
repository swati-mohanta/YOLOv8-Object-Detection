# 🚗 YOLOv8 Object Detection on Self-Driving Cars Dataset

This project focuses on **Object Detection** using **YOLOv8 (You Only Look Once version 8)** applied to the **Self-Driving Cars Dataset**.  
The aim is to accurately detect vehicles, pedestrians, and other driving-related objects from images — simulating the perception systems used in autonomous driving.

Implemented in **Google Colab**, this experiment demonstrates the complete workflow: data preparation, model training, evaluation, and visualization of detection results.

---

## Project Overview

The objective of this project is to train a YOLOv8 model capable of detecting various objects typically seen in driving scenarios.  
The **Self-Driving Cars Dataset** contains labeled images captured from a car’s perspective, making it ideal for experimenting with real-world object detection in traffic environments.

### 🔍 Steps Performed
1. **Environment Setup** – Installed required libraries and configured YOLOv8.  
2. **Dataset Integration** – Loaded and prepared the Self-Driving Cars dataset.  
3. **Training Phase** – Trained YOLOv8 on the dataset with custom parameters.  
4. **Validation & Testing** – Evaluated model performance on unseen samples.  
5. **Inference & Visualization** – Generated predictions with bounding boxes and confidence scores.

---

## Tech Stack

- **Language:** Python  
- **Framework:** PyTorch  
- **Model:** YOLOv8 (Ultralytics)  
- **Environment:** Google Colab  
- **Libraries:** `ultralytics`, `opencv-python`, `matplotlib`, `numpy`, `torch`

---

## Dataset Details

Dataset: **[Self-Driving Cars Dataset](https://www.kaggle.com/datasets/alincijov/self-driving-cars)**
This dataset contains labeled images for detecting objects relevant to self-driving systems such as:
- Cars
- Pedestrians
- Road signs
- Bicycles and other road entities
Each image includes bounding boxes and class labels compatible with YOLO format.

---

## Experiment Details

- **Model Used:** YOLOv8n / YOLOv8s
- **Dataset:** Self-Driving Cars Dataset
- **Epochs:** Variable (based on tuning and GPU availability)
- **Metrics Evaluated:** Precision, Recall, mAP (mean Average Precision)
- **Output:** Bounding boxes with class names and confidence scores
The trained YOLOv8 model effectively identifies and classifies multiple object categories in traffic scenes with strong detection accuracy.
