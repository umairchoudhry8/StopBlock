# StopBlock

The StopBlock is a Python application designed to monitor and assess the condition of culverts using computer vision techniques. It captures images of culverts, performs object detection to identify openings, and classifies these openings to determine if the culvert is blocked or clear. This README provides an overview of the system, its components, and instructions for setting it up and running it.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [License](#license)

## Introduction

The Culvert Monitoring System comprises several components and functionalities listed as follows:

- **Camera Initialization**: The system initializes and powers on the camera for capturing images of the culvert.

- **Image Capture**: It captures an image of the culvert using the initialized camera.

- **Object Detection**: The captured image is processed using an object detection model (ONNX) to identify openings in the culvert.

- **Image Classification**: For each detected opening, the system further classifies it using an image classification model (ONNX) to determine if it is clear, partially blocked, or blocked.

- **Blockage Statistics**: The system calculates blockage statistics based on the classified openings to assess the condition of the culvert.

- **MQTT Communication**: It communicates the results, including the number of openings, clear openings, partially blocked openings, blocked openings, scores from the object detection model, and classifications, to an MQTT broker.

## Prerequisites

Before setting up and running the AtopBlock application, ensure you have the following prerequisites:

- NVIDIA Jetson device (e.g., Jetson Nano) with Ubuntu-based OS
- Python 3.x installed
- OpenCV (`cv2`) library
- TensorFlow (`tensorflow`) library
- ONNX Runtime (`onnxruntime`) library
- Uptime (`uptime`) library
- MQTT library (e.g., Paho MQTT)
- Access to a camera compatible with your Jetson device
- Configuration file (`config.json`) with necessary parameters (see Configuration section)

## Setup
- Navigate to the project directory
- Install the required Python libraries
- Place the ONNX model files (frcnn_resnet50.onnx and ResNet50_Best.onnx) in the project directory.
- Create an output directory in the project directory to store captured images and blockage statistics

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute it as per the terms of the license.

