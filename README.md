# Object Detection App (YOLOv5)

A simple Python object detection app built on the pretrained YOLOv5 model from Ultralytics, with support for both static images and live webcam feeds.

## Features

- Detect objects in any image file
- Real-time object detection from a webcam stream
- Press `q` to quit the webcam view

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- NumPy

```bash
pip install torch opencv-python numpy
```

## Usage

```bash
python app.py
```

You will be prompted to choose a mode:

1. **Detect objects in an image** — enter the path to an image file
2. **Real-time object detection from webcam** — uses your default camera

The pretrained `yolov5s.pt` weights are included in this repository, and `test_image.jpg` is provided as a sample input.

## How it works

The model is loaded through `torch.hub` from the `ultralytics/yolov5` repository using the lightweight `yolov5s` architecture. Frames or images are passed to the model, and detections are rendered and displayed with OpenCV.
