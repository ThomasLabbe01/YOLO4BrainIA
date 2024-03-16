import cv2
import os
import time
from multiprocessing import Pool
import colorsys
import json

# THE WORKERS FOR THE IMWRITE MULTIPROCESS
def extract_frames_worker(args):
    frame, frame_path, quality = args
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_WEBP_QUALITY, quality])


class FrameExtractor:
    supported_resolutions = {
        "livrable1": (1536, 1536),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "2k": (2560, 1440),
    }

    def __init__(self, DETECTOR, classes, resolution="livrable1"):
        self.DETECTOR = DETECTOR
        self.USE_MULTIPROCESSING = True
        self.USE_FULL_LENGTH = False  # TOGGLE FULL VIDEO LENGTH EXTRACTION
        self.MAX_NUMBER_OF_FRAME = 50
        if resolution not in self.supported_resolutions:
            raise ValueError(
                f"Invalid resolution: {resolution}. Supported resolutions are: {self.supported_resolutions.keys()}"
            )
        self.resolution = self.supported_resolutions[resolution]
        self.classes = classes
        self.final_dict = {}
        self.init_data()

    def init_data(self):
        self.check_if_Raph()
        self.final_dict["classes"] = {}
        self.prepare_colors()
        self.prepare_states()
        # self.final_dict = {"classes": {"eau": "#0008ff", "lit": "#fb00ff", "mangeoire": "#ff9500", "mouse": "#FF0000", "nid": "#0000FF", "roulette": "#00FF00"}}

    def check_if_Raph(self):
        dev_name_dir = os.path.expandvars("%USERPROFILE%")
        dev_name_index = dev_name_dir.rfind("\\")
        dev_name = dev_name_dir[dev_name_index + 1 :]
        if dev_name == "rapha" or dev_name == "avid":
            self.MAX_NUMBER_OF_FRAME = 10

    def get_frame_aspect_ratio(self, frame):
        return frame.shape[0] / frame.shape[1]

    def write_config_json_file(self, directory, frame_name):

        file_name = os.path.splitext(frame_name)[0]
        extension = os.path.splitext(frame_name)[1]
        file_path = os.path.join(directory, file_name, file_name + "_config.json")
        try:
            with open(file_path, 'w') as json_file:
                json.dump(self.final_dict, json_file, indent=4)
            print("Config File was created with success")
        except:
            print("ERROR CREATING CONFIG FILE")


    def extract_frames(self, fileName, workspacePath):

        frame_count = 0
        frame_list = []
        frame_batch_size = 0
        file_path, unlabelled_path, labelled_path = self.create_folder_structure(fileName, workspacePath)
        capture = cv2.VideoCapture(file_path)
        video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.USE_FULL_LENGTH:
            self.MAX_NUMBER_OF_FRAME = video_length
        print("ORIGINAL VIDEO HAS", video_length, "FRAMES")
        print("EXTRACTING", self.MAX_NUMBER_OF_FRAME, "FRAMES")
        self.final_dict["numOfFrames"] = self.MAX_NUMBER_OF_FRAME
        self.write_config_json_file(workspacePath, fileName)

        if self.USE_MULTIPROCESSING:
            AVAILABLE_CPU = os.cpu_count() - 1
            frame_batch_size = AVAILABLE_CPU
            pool = Pool(processes=AVAILABLE_CPU)
            print("NUMBER OF PROCESS", AVAILABLE_CPU)
        else:
            print("NUMBER OF PROCESS 1 - NO MULTIPROCESSING")

        self.start_time = time.time()
        remaining_frames = self.MAX_NUMBER_OF_FRAME

        # while capture.isOpened():
        while frame_count < self.MAX_NUMBER_OF_FRAME:
            success, frame = capture.read()

            if not success:
                break

            if frame.shape[:2] != self.resolution:
                aspect_ratio = self.get_frame_aspect_ratio(frame)
                frame = cv2.resize(frame, (self.resolution[1], int(self.resolution[0] * aspect_ratio)))

            frame_filename = f"frame{frame_count}.webp"
            frame_path_1 = os.path.join(unlabelled_path, frame_filename)
            frame_path_2 = os.path.join(labelled_path, frame_filename)
            # if os.path.exists(frame_path_1) or os.path.exists(frame_path_2):
            #     print(f"Frame {frame_count} already exists in one of the paths. Skipping...")
            #     frame_count += 1
            #     continue

            if self.USE_MULTIPROCESSING:
                remaining_frames -= 1
                frame_list.append([frame_count, frame])
                if len(frame_list) == frame_batch_size:
                    start = time.time()
                    # Process a batch of frames using multiprocessing
                    batch_args = [(data[1], os.path.join(unlabelled_path, f"frame{data[0]}.webp"), 75) for data in frame_list]
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
            # Perform detection and tracking for the frame
            start = time.time()
            detections = self.DETECTOR(frame)
            time2 = time.time() - start
            self.time2_list.append(time2)  # Append time2 to the list

            start = time.time()
            tracking_output = TRACKER.update(detections[0]).to_dict()

            self.add_state_to_output(tracking_output)
            time3 = time.time() - start
            self.time3_list.append(time3)  # Append time3 to the list


            # VERSION BATCHES FRAME
            # frame_list.append(frame)
            # if len(frame_list) == frame_batch_size:
            #     start = time.time()
            #     list_detections = DETECTOR(frame_list)
            #     time2 = time.time() - start
            #     time2_list.append(time2)  # Append time2 to the list
            #
            #     start = time.time()
            #     for detections in list_detections:
            #         start = time.time()
            #         frame_name = "frame" + str(temp_frame_number)
            #         tracking_output = TRACKER.update(detections).to_dict(frame_name)
            #         temp_tracking = copy.deepcopy(tracking_output)
            #         tracking_output_with_states = convert_detections(frame_name, temp_tracking)
            #         final_dict.update(tracking_output_with_states)
            #         temp_frame_number += 1
            #     time3 = time.time() - start
            #     time3_list.append(time3)  # Append time3 to the list
            #
            #     frame_list.clear()

            # SMALL FIX FOR FRONT END DATA STRUCTURE
            if "mangeoir" in tracking_output[frame_name]["data"]:
                tracking_output[frame_name]["data"]["mangeoire"] = tracking_output[
                    frame_name
                ]["data"].pop("mangeoir")

            # self.final_dict.update(tracking_output)
            video_name = os.path.splitext(fileName)[0]
            with open(
                os.path.join(
                    workspacePath,
                    video_name,
                    "jsonFolder",
                    f"{os.path.splitext(fileName)[0]}_{frame_name}.json",
                ),
                "w",
            ) as outfile:
                json.dump(tracking_output, outfile)

            self.print_progress_bar(frame_count, self.MAX_NUMBER_OF_FRAME)
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

    def createDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def resize_frame_with_aspect_ratio(self, frame, target_size):
        height, width, _ = frame.shape
        aspect_ratio = width / float(height)

        if width > target_size or height > target_size:
            if width > height:
                new_width = target_size
                new_height = int(target_size / aspect_ratio)
            else:
                new_height = target_size
                new_width = int(target_size * aspect_ratio)

            resized_frame = cv2.resize(frame, (new_width, new_height))
            return resized_frame
        else:
            return frame

    def create_folder_structure(self, fileName, workspacePath):
        file_path = os.path.join(workspacePath, fileName)
        folder_name = os.path.splitext(fileName)[0]

        folder_path = os.path.join(workspacePath, folder_name)
        self.createDir(folder_path)

        jsonData_path = os.path.join(folder_path, "jsonFolder")
        self.createDir(jsonData_path)

        unlabelled_path = os.path.join(folder_path, "unlabelled")
        self.createDir(unlabelled_path)

        labelled_path = os.path.join(folder_path, "labelled")
        self.createDir(labelled_path)

        train_path = os.path.join(folder_path, "labelled", "train")
        train_img_path = os.path.join(folder_path, "labelled", "train", "images")
        train_labels_path = os.path.join(folder_path, "labelled", "train", "labels")
        self.createDir(train_path)
        self.createDir(train_img_path)
        self.createDir(train_labels_path)

        valid_path = os.path.join(folder_path, "labelled", "val")
        valid_img_path = os.path.join(folder_path, "labelled", "val", "images")
        valid_labels_path = os.path.join(folder_path, "labelled", "val", "labels")

        self.createDir(valid_path)
        self.createDir(valid_img_path)
        self.createDir(valid_labels_path)
        return file_path, unlabelled_path, labelled_path

    def add_state_to_output(self, output):
        for frame in output:
            for info_key, info in output[frame].items():
                if info_key == "data":
                    for class_data in info.values():
                        for class_data_key, object_data in class_data.items():
                            if class_data_key.isdigit():
                                self.format_object_data(object_data)
                            elif class_data_key == "unconfirmed_tracks":
                                for unconfirmed_track_id in object_data:
                                    self.format_object_data(
                                        object_data[unconfirmed_track_id]
                                    )

    def format_object_data(self, object_data):
        if "bbox" in object_data:
            object_data["source"] = "AI"
            object_data["state"] = "alive"
            object_data["bbox"] = str(
                [
                    int(float(x) * self.resolution[0])
                    if i % 2 == 0
                    else int(float(x) * self.resolution[1])
                    for i, x in enumerate(object_data["bbox"])
                ]
            )

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
            self.print_average_times(self.time3_list, "Tracker")

    def print_average_times(self, time_list, label):
        if time_list:
            avg_time = sum(time_list) / len(time_list)
            print(f"Average Time for {label}: {avg_time:.4f} seconds")
        else:
            print(f"No data available for {label}")

    def format_time(self, seconds):
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    def print_progress_bar(self, iteration, total, bar_length=50):
        progress = iteration / total
        arrow = "=" * int(round(bar_length * progress))
        spaces = " " * (bar_length - len(arrow))
        progress_bar = f"[{arrow}{spaces}] {int(progress * 100)}%"

        if iteration > 0:
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time / progress
            estimated_remaining_time = estimated_total_time - elapsed_time
            progress_bar += f" - {self.format_time(estimated_remaining_time)} remaining"

        print("\r" + progress_bar, end="")

    def prepare_colors(self):
        color_list, fade_color_list = self.generate_color_list(len(self.DETECTOR.supported_classes))
        self.final_dict["classes"] = {}
        self.final_dict["classesFaded"] = {}

        for id, class_name in enumerate(self.DETECTOR.supported_classes):
            if class_name == "mangeoir":
                temp_class_type = "mangeoire"
            else:
                temp_class_type = class_name
            self.final_dict["classes"][temp_class_type] = color_list[id]
            self.final_dict["classesFaded"][temp_class_type] = fade_color_list[id]


    def prepare_states(self):
        state_list = ["Exploring",
        "Investigating",
        "Fighting",
        "Grooming",
        "Rearing",
        "Grooming+",
        "Running",
        "Resting",
        "Other",
        "No State"]
        self.final_dict["states"] = {}
        for item in state_list:
            self.final_dict["states"][item] = {}


    def generate_color_list(self, num_colors):
        # Define constant saturation and value (brightness)
        saturation = 1.0
        saturation_low = 0.2
        value = 0.8
        # Generate a list of evenly spaced hue values
        hue_values = [i / num_colors for i in range(num_colors)]
        # Initialize an empty list to store the color codes
        color_list = []
        fade_color_list = []
        # Convert HSV to RGB and create color codes
        for hue in hue_values:
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert RGB to HEX format
            hex_color = "#{:02X}{:02X}{:02X}".format(
                int(rgb_color[0] * 255),
                int(rgb_color[1] * 255),
                int(rgb_color[2] * 255),
            )
            color_list.append(hex_color)

            rgb_color = colorsys.hsv_to_rgb(hue, saturation_low, value)
            # Convert RGB to HEX format
            hex_color = "#{:02X}{:02X}{:02X}".format(
                int(rgb_color[0] * 255),
                int(rgb_color[1] * 255),
                int(rgb_color[2] * 255),
            )
            fade_color_list.append(hex_color)

        return color_list, fade_color_list
