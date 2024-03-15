"""
This script uses the YOLOv8 model to perform object detection on images from a specified folder. 
The script loads the YOLO model, sets a confidence threshold, and detects objects from selected classes. 
The selected classes are specified using class indices. The results are saved, including both annotated images and text files
with object coordinates. Make sure to adjust the model path, selected_classes_indices, and image_path based on your preferences.

This python file is used as a way to do active training using larger yolo models. Active training enables us to mix images from different databases
Example : 
we have data from coco dataset and openimages dataset
In coco dataset, "person" is labeled, and in openimages, "human hand" is labeled
the "human hand" object is not labeled in cocodataset, and vis versa
We use the variable selected_classes_indices_xx to predict the desired class from one dataset to the other
We use modelA to detect classesA in dataset B, and modelB to detect classesB in datasetA
This will reduce false positive during training

Requirements:
- Install Ultralytics library: pip install ultralytics
- Ensure the YOLO model path ("yolov8x.pt") is correct.
- Modify the selected_classes_indices and image_path according to your specific use case.
"""
from ultralytics import YOLO

# Load the YOLOv8 model
model_coco = YOLO("yolov8x.pt")
model_open = YOLO("yolov8x-oiv7.pt")

# Set the confidence threshold
conf_thresh = 0.5

# Specify the selected classes for detection
selected_classes_indices_coco = [47, 24, 46, 39, 45, 67, 56, 41, 42, 66, 43, 63, 64, 49, 65, 44, 28, 77, 27, 25]
selected_classes_indices_openimages = [267, 165, 56, 218, 476, 115, 54]

# Specify the path to the folder containing images for detection
image_path_coco_train = "C:/Users/Thoma/ProjetYOLO/Yolo/trained_model_v5/dataformattedopenimages/validation/images"
image_path_coco_val = "C:/Users/Thoma/ProjetYOLO/Yolo/trained_model_v5/dataformattedopenimages/validation/images"
image_path_open_train = "C:/Users/Thoma/ProjetYOLO/Yolo/trained_model_v5/dataformattedopenimages/validation/images"
image_path_open_val = "C:/Users/Thoma/ProjetYOLO/Yolo/trained_model_v5/dataformattedopenimages/validation/images"

# Predict images from
print("Using best yolo model trained on coco to create annotations on the open images dataset (train)") 
results = model_coco.predict(source=image_path_open_train, save=True, classes=selected_classes_indices_coco, conf=conf_thresh, save_txt=True)

print("Using best yolo model trained on coco to create annotations on the open images dataset (validation)") 
results = model_coco.predict(source=image_path_open_val, save=True, classes=selected_classes_indices_coco, conf=conf_thresh, save_txt=True)

print("Using best yolo model trained on openImages to create annotations on the coco  dataset (train)") 
results = model_open.predict(source=image_path_coco_train, save=True, classes=selected_classes_indices_openimages, conf=conf_thresh, save_txt=True)

print("Using best yolo model trained on openImages to create annotations on the coco dataset (validation)") 
results = model_open.predict(source=image_path_coco_val, save=True, classes=selected_classes_indices_openimages, conf=conf_thresh, save_txt=True)

print("predictions completed. labels can be found in the four last 'predictxx' folder in C:/Users/Thoma/runs/detect")