import os
from ultralytics import YOLO
import cv2
import os
import time
from multiprocessing import Pool
import colorsys
import json
from data_config import *
import numpy as np


# THE WORKERS FOR THE IMWRITE MULTIPROCESS
def extract_frames_worker(args):
    frame, frame_path, quality = args
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_WEBP_QUALITY, quality])


class Predictor:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model_custom = YOLO(self.weights_path)
        self.classes = load_yaml_file(os.path.join(os.path.dirname(os.path.dirname(weights_path)), "combined_config.yaml"))["names"]
        self.final_dict = {"classes": {}}
        self.final_dict = prepare_colors(self.final_dict, self.classes)
        self.final_dict = prepare_states(self.final_dict)

        #SETTINGS
        self.conf_threshold = 0.05
        self.MAX_NUMBER_OF_FRAME = 50
        self.USE_FULL_LENGTH = False
        self.USE_MULTIPROCESSING = True

        #FOR BENCHMARKING
        self.start_time = 0
        self.time1_list = []
        self.time2_list = []
        self.time3_list = []

    def predict(self, img, output_dir='output'):
        model_tensor = self.model_custom.predict(source=img, conf=self.conf_threshold,
                                                 save=False, save_txt=False, verbose=False)
        bbox_coords_list = model_tensor[0].boxes.xyxy.cpu().numpy()
        bbox_names = model_tensor[0].boxes.cls.cpu().numpy()
        confidences = model_tensor[0].boxes.conf.cpu().numpy()

        obj_quantity = {}
        unique_values, counts = np.unique(bbox_names, return_counts=True)
        obj_quantity = dict(zip(unique_values, counts))

        predictions = {}
        for index, id in enumerate(bbox_names):
            bbox_coords = bbox_coords_list[int(index)]
            int_list = [int(x) for x in bbox_coords]
            class_label = self.classes[id][0:-1]
            confidence = confidences[index]

            if class_label not in predictions:
                predictions[class_label] = {}

            if obj_quantity[id] > 1:
                obj_id = len(predictions[class_label])
            else:
                obj_id = 0

            predictions[class_label][obj_id] = {"bbox": str(int_list), "confidence": float(confidence)}

        ## TO DISPLAY IMAGES
        #     object_color = hex_to_rgb(self.final_dict["classes"][class_label])
        #     x_min, y_min, x_max, y_max = bbox_coords
        #     cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), object_color, 2)
        #     # cv2.putText(img, class_label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 2)
        # # Save the image with bounding boxes drawn
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # output_path = os.path.join(output_dir, 'output_image' + str(int(x_min)) + '.jpg')
        # cv2.imwrite(output_path, img)

        return predictions

    def extract_frames(self, fileName, workspacePath):
        frame_count = 0
        frame_list = []
        frame_batch_size = 0
        file_path, unlabelled_path, labelled_path = create_folder_structure(fileName, workspacePath)
        capture = cv2.VideoCapture(file_path)
        video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.USE_FULL_LENGTH:
            self.MAX_NUMBER_OF_FRAME = video_length
        print("ORIGINAL VIDEO HAS", video_length, "FRAMES")
        print("EXTRACTING", self.MAX_NUMBER_OF_FRAME, "FRAMES")
        self.final_dict["numOfFrames"] = self.MAX_NUMBER_OF_FRAME
        write_config_json_file(self.final_dict, workspacePath, fileName)

        if self.USE_MULTIPROCESSING:
            AVAILABLE_CPU = os.cpu_count() - 1
            frame_batch_size = AVAILABLE_CPU
            pool = Pool(processes=AVAILABLE_CPU)
            print("NUMBER OF PROCESS", AVAILABLE_CPU)
        else:
            print("NUMBER OF PROCESS 1 - NO MULTIPROCESSING")

        self.start_time = time.time()
        remaining_frames = self.MAX_NUMBER_OF_FRAME

        while frame_count < self.MAX_NUMBER_OF_FRAME:
            # Read the frame
            success, frame = capture.read()
            if not success:
                break

            #TODO CHECK IF FRAME RESIZE IS NECESSARY
            frame_filename = f"frame{frame_count}.webp"
            frame_path_1 = os.path.join(unlabelled_path, frame_filename)
            frame_path_2 = os.path.join(labelled_path, frame_filename)
            # if os.path.exists(frame_path_1) or os.path.exists(frame_path_2):
            #     print(f"Frame {frame_count} already exists in one of the paths. Skipping...")
            #     frame_count += 1
            #     continue

            # Write the frame local storage using MULTIPROCESSING or SINGLEPROCESSING
            if self.USE_MULTIPROCESSING:
                remaining_frames -= 1
                frame_list.append([frame_count, frame])
                if len(frame_list) == frame_batch_size:
                    start = time.time()
                    # Process a batch of frames using multiprocessing
                    batch_args = [(data[1], os.path.join(unlabelled_path, f"frame{data[0]}.webp"), 75) for data in
                                  frame_list]
                    pool.map(extract_frames_worker, batch_args)
                    stop = time.time()
                    time1 = stop - start
                    self.time1_list.extend(
                        [time1] * frame_batch_size
                    )  # Append time1 to the list
                    frame_list.clear()

            else:
                start = time.time()
                cv2.imwrite(frame_path_1, frame, [cv2.IMWRITE_WEBP_QUALITY, 75])
                stop = time.time()
                time1 = stop - start
                self.time1_list.append(time1)  # Append time1 to the list

            # Process the remaining frames using multiprocessing
            if remaining_frames == 0 and self.USE_MULTIPROCESSING:
                pool = Pool(processes=len(frame_list))
                batch_args = [
                    (data[1], os.path.join(unlabelled_path, f"frame{data[0]}.webp"), 75)
                    for data in frame_list
                ]
                pool.map(extract_frames_worker, batch_args)
                pool.close()
                pool.join()

            frame_name = f"frame{frame_count}"
            # Perform detection for the frame
            start = time.time()
            tracking_output = self.predict(frame)
            json_output = {frame_name: {}}
            json_output[frame_name]["image_size"] = [frame.shape[1], frame.shape[0]]
            json_output[frame_name]["data"] = tracking_output
            time2 = time.time() - start
            self.time2_list.append(time2)  # Append time2 to the list

            video_name = os.path.splitext(fileName)[0]
            json_path = os.path.join(workspacePath, video_name, "jsonFolder", f"{os.path.splitext(fileName)[0]}_{frame_name}.json")
            with open(json_path, "w") as outfile:
                json.dump(json_output, outfile)

            print_progress_bar(self.start_time, frame_count, self.MAX_NUMBER_OF_FRAME)
            # IF progress bar is not fun for you, use this instead
            # if frame_count % 10 == 0:
            #     print(f"Frame {frame_count} extracted.")

            frame_count += 1

        if self.USE_MULTIPROCESSING:
            pool.close()
            pool.join()

        capture.release()
        total_time = time.time() - self.start_time

        self.print_final_timer_score(total_time)

        return self.final_dict

    def print_final_timer_score(self, total_time):
        print("\nTotal Time for Extraction is:", total_time)
        if (
            len(self.time1_list) != 0
            and len(self.time2_list) != 0
            and len(self.time3_list) != 0
        ):
            # Calculate and print the average times
            self.print_average_times(self.time1_list, "Writing")
            self.print_average_times(self.time2_list, "Detector")

    def print_average_times(self, time_list, label):
        if time_list:
            avg_time = sum(time_list) / len(time_list)
            print(f"Average Time for {label}: {avg_time:.4f} seconds")
        else:
            print(f"No data available for {label}")


if __name__ == "__main__":
    fileName = "CustomModelSmall.pt"
    script_directory = os.path.dirname(os.path.abspath(__file__))
    repo_directory = os.path.dirname(script_directory)
    weights_path = os.path.join(repo_directory, "weights", fileName)
    PREDICTOR = Predictor(weights_path)

    folder_path = "C:/Users/david/OneDrive - Université Laval/GLO-7030 Shared Folder/videos"
    video_path = os.path.join(folder_path, "IMG_3342.MOV")
    # folder_path = "C:/Users/david/Desktop"
    # video_path = os.path.join(folder_path, "testing.JPG")
    extension, base_name, frame_name, directory = split_path(video_path)
    predictions = PREDICTOR.extract_frames(base_name, directory)
