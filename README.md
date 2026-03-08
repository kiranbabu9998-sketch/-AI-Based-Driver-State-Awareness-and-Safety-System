# AI-Based Driver Drowsiness Detection System

A real-time deep learning system for detecting driver drowsiness using computer vision.
The system uses a **YOLO-based deep learning model** to monitor driver facial features such as eye closure and yawning and triggers a **voice alert system** to warn the driver when fatigue is detected.

The project includes a **Flask-based web interface**, a trained object detection model, and a real-time video processing pipeline.

---

# Table of Contents

* Overview
* Project Structure
* Features
* Model
* System Workflow
* Installation
* Running the Application
* Supported Inputs
* Future Improvements
* Dependencies

---


## Overview

Driver fatigue is one of the leading causes of road accidents worldwide. Long driving hours and reduced alertness can significantly affect a driver's reaction time and decision-making ability, increasing the risk of accidents.This project proposes an **AI-powered Driver Monitoring System (DMS)** that continuously observes the driver using computer vision and deep learning techniques. The system analyzes facial cues such as **eye closure and yawning** in real time using a trained YOLO-based model.

Currently, the system focuses on **driver monitoring and vehicle safety response within a single vehicle environment**. Communication with other nearby vehicles (Vehicle-to-Vehicle communication) is not implemented in the present version.This project demonstrates how **AI-based driver monitoring systems can improve road safety by detecting fatigue early and initiating preventive safety actions**.


The system detects:

* Eye closure
* Yawning
* Signs of driver fatigue

When drowsiness is detected, the system immediately triggers an **audio alert** to wake the driver and prevent accidents.

---

## Project Structure

```
Driver-Drowsiness-Detection/
│
├── Backend
│   └── app.py              # Flask server handling requests
│
├── Detection
│   └── run.py              # Real-time drowsiness detection pipeline
│
├── Frontend
│   └── index.html          # Web interface for monitoring
│
├── Model
│   └── drowsy.pt           # YOLO trained model
│
└── Assets
    └── voice_driver.mp3    # Audio alert for driver warning
```

---

# Features

## Real-Time Driver Monitoring

The system captures live video frames and processes them using a deep learning model to monitor the driver's facial behavior.

## Eye Closure Detection

The model detects whether the driver's eyes are closed for a prolonged duration, which is a key indicator of fatigue.

## Yawning Detection

Yawning is another strong signal of drowsiness. The system detects mouth opening patterns to identify yawning events.

## Voice Alert System

If drowsiness is detected, an **audio warning is played automatically** to alert the driver.

## Web Interface

A simple web interface allows the user to start the detection system and visualize the monitoring process.

---

# Model

| Property   | Detail                            |
| ---------- | --------------------------------- |
| Model Type | YOLO-based Object Detection       |
| Framework  | PyTorch / Ultralytics             |
| Input      | Real-time camera frames           |
| Output     | Detection of eye and mouth states |
| Purpose    | Detect driver fatigue indicators  |
| Model File | `drowsy.pt`                       |

The trained model detects facial states and classifies them into categories such as:

* Eyes Open
* Eyes Closed
* Yawning
* Alert State

---

# System Workflow

1. Capture real-time video from the driver's camera.
2. Extract frames using OpenCV.
3. Pass frames to the YOLO model (`drowsy.pt`).
4. Detect eye and mouth states.
5. Analyze driver behavior patterns.
6. If drowsiness is detected:

   * Trigger an **audio alert**
   * Display warning on the interface.

---

# Installation

## Prerequisites

Python ≥ 3.11 recommended

Install required libraries:

pip install ultralytics
pip install opencv-python
pip install flask
pip install numpy

---

# Running the Application

click open in terminal where you save all the above files there you type below:

python app.py

After running , open link in end:

e.g: http://localhost:5000

The system will start monitoring the driver using the camera.

---

# Supported Inputs

| Input                         | Description                 |
| ----------------------------- | --------------------------- |
| Webcam Video                  | Real-time driver monitoring |
| Pre-recorded video (optional) | Can be used for testing     |

---

# Future Improvements

* Integration with **Vehicle-to-Vehicle (V2V) communication**
* Automatic **vehicle speed reduction when drowsiness detected**
* Mobile application for remote monitoring
* Cloud-based driver behavior analytics
* Integration with autonomous vehicle safety systems

---

# Dependencies

| Package          | Purpose                      |
| ---------------- | ---------------------------- |
| Flask            | Web server for the interface |
| OpenCV           | Video processing             |
| Ultralytics YOLO | Object detection model       |
| NumPy            | Numerical computation        |

---

# Conclusion

This project demonstrates how **Artificial Intelligence and Computer Vision** can be used to improve road safety by detecting driver drowsiness in real time.

Such intelligent monitoring systems can help reduce accidents and enhance the safety of modern transportation systems.

## License

This project uses materials licensed under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license.






