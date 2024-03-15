"""
This script uses the Ultralytics YOLO library to train a YOLOv8 object detection model on a custom dataset. 
It loads the model configuration from the specified YAML file, initializes the model with or without
pretrained weights, and resumes training if specified. The training parameters such as the number of
epochs, learning rate, and training device are set in the `model.train()` method.
Modify the `config_path`, `model_path`, and `resume` variables as needed for your dataset and training preferences.
"""
 

from ultralytics import YOLO
# Load a model 
config_path = "C:/Users/Thoma/OneDrive/Documents/data_yolo_training_v2/dataset.yaml" # path to data
#model_path = "C:/Users/Thoma/runs/detect/train91/weights/best.pt" # path to weights. .yaml from scratch, .pt from pretrained weights
model_path = "yolov8s.pt"
resume = False # if the training stopped at any point, use this to keep going

if __name__ == '__main__':
        model = YOLO(model_path)  # build a new model from scratch 
        results = model.train(data=config_path, epochs=400, resume=resume, lr0=0.00269, lrf=0.00288, momentum=0.73375, weight_decay=0.00015, warmup_epochs=1.22935, warmup_momentum=0.1525, box=18.27875, cls=1.32899, dfl=0.56016, hsv_h=0.01148, hsv_s=0.53554, hsv_v=0.13636, degrees=0, translate=0.12431, scale=0.07643, shear=0, perspective=0, flipud=0.0, fliplr=0.08631, mosaic=0.42551, mixup=0, copy_paste=0, device=0)  # train the model
   
