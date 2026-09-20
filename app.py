import torch
import cv2
import numpy as np

# Load the pre-trained YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def detect_objects(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    results.show()  # Display detected objects

def detect_from_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        results = model(frame)

        # Convert results to OpenCV format
        rendered_img = np.array(results.render()[0])
        cv2.imshow("Real-Time Object Detection", rendered_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Choose mode
print("Choose an option:\n1. Detect objects in an image\n2. Real-time object detection from webcam")
choice = input("Enter 1 or 2: ")

if choice == "1":
    image_path = input("Enter image path: ")
    detect_objects(image_path)
elif choice == "2":
    detect_from_webcam()
else:
    print("Invalid choice. Exiting.")


