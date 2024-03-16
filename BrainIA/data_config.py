import yaml
import colorsys
import os
import cv2
import json
import time

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None


def prepare_colors(final_dict, classes):
    num_classes = len(classes)
    color_list, fade_color_list = generate_color_list(num_classes)
    final_dict["classes"] = {}
    final_dict["classesFaded"] = {}
    for class_index in classes:
        temp_class_type = classes[class_index][0:-1]

        final_dict["classes"][temp_class_type] = color_list[class_index]
        final_dict["classesFaded"][temp_class_type] = fade_color_list[class_index]
    return final_dict

def prepare_states(final_dict):
    state_list = ["No State"]
    final_dict["states"] = {}
    for item in state_list:
        final_dict["states"][item] = {}
    return final_dict

def generate_color_list(num_colors):
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

def write_config_json_file(final_dict, directory, frame_name):

    file_name = os.path.splitext(frame_name)[0]
    extension = os.path.splitext(frame_name)[1]
    file_path = os.path.join(directory, file_name, file_name + "_config.json")
    try:
        with open(file_path, 'w') as json_file:
            json.dump(final_dict, json_file, indent=4)
        print("Config File was created with success")
    except:
        print("ERROR CREATING CONFIG FILE")

def create_folder_structure(fileName, workspacePath):
    file_path = os.path.join(workspacePath, fileName)
    folder_name = os.path.splitext(fileName)[0]

    folder_path = os.path.join(workspacePath, folder_name)
    createDir(folder_path)

    jsonData_path = os.path.join(folder_path, "jsonFolder")
    createDir(jsonData_path)

    unlabelled_path = os.path.join(folder_path, "unlabelled")
    createDir(unlabelled_path)

    labelled_path = os.path.join(folder_path, "labelled")
    createDir(labelled_path)

    train_path = os.path.join(folder_path, "labelled", "train")
    train_img_path = os.path.join(folder_path, "labelled", "train", "images")
    train_labels_path = os.path.join(folder_path, "labelled", "train", "labels")
    createDir(train_path)
    createDir(train_img_path)
    createDir(train_labels_path)

    valid_path = os.path.join(folder_path, "labelled", "val")
    valid_img_path = os.path.join(folder_path, "labelled", "val", "images")
    valid_labels_path = os.path.join(folder_path, "labelled", "val", "labels")

    createDir(valid_path)
    createDir(valid_img_path)
    createDir(valid_labels_path)
    return file_path, unlabelled_path, labelled_path

def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_frame_with_aspect_ratio(frame, target_size):
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


def format_time(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def print_progress_bar(start_time, iteration, total, bar_length=50):
    progress = iteration / total
    arrow = "=" * int(round(bar_length * progress))
    spaces = " " * (bar_length - len(arrow))
    progress_bar = f"[{arrow}{spaces}] {int(progress * 100)}%"

    if iteration > 0:
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress
        estimated_remaining_time = estimated_total_time - elapsed_time
        progress_bar += f" - {format_time(estimated_remaining_time)} remaining"

    print("\r" + progress_bar, end="")


