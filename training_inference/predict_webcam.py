import os
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import time

def draw_boxes(frame, results, threshold, threshold_value):
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if threshold and score <= threshold_value:
            continue

        accuracy_text = f"{score:.2f}"
        label_text = f"{results.names[int(class_id)].upper()} {accuracy_text}"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, label_text, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    return frame

# Load the YOLOv8 model
model_path = "C:/Users/Thoma/ProjetYOLO/Yolo/VisionFusionProject/models/CustomModelSmall.pt"
model = YOLO(model_path)

threshold = True
threshold_value = 0.5

# URL for the webcam stream from your phone
url = "http://10.0.0.91:8080/shot.jpg"

# Get the frame dimensions
width, height = 640, 480  # You can adjust these values based on your preferences

# Define the codec and create VideoWriter object
output_path = os.path.join('C:/Users/Thoma/ProjetYOLO/Yolo/trained_model_v5/videosOuput', 'testwebcam.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

inference_times = []
all_times = []
while True:
    start_time = time.time()
    # Capture image from the webcam
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    frame = cv2.resize(frame, (width, height))

    # Run YOLOv8 inference on the frame
    s_inference_time = cv2.getTickCount()
    results = model(frame)[0]
    e_inference_time = cv2.getTickCount()
    
    inference_time_ms = (e_inference_time - s_inference_time) / cv2.getTickFrequency() * 1000.0
    inference_times.append(inference_time_ms)
    end_time = time.time()
    total_time = (end_time-start_time)*1000
    all_times.append(total_time)

    # Draw boxes on the frame based on threshold parameters
    frame = draw_boxes(frame, results, threshold, threshold_value)

    # Calculate the mean inference time
    mean_inference_time = np.mean(inference_times[-10:])
    mean_all_time = np.mean(all_times[-10:])
    fps = f"Yolo process: {mean_inference_time:.2f} ms"
    AllProcess = f"All process: {mean_all_time:.2f} ms"

    # Calculate the entire process time
    cv2.putText(frame, fps, (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, AllProcess, (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow("YOLOv8 Inference", frame)

    # Write the frame with annotations to the output video
    out.write(frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the video writer and close windows
out.release()
cv2.destroyAllWindows()
