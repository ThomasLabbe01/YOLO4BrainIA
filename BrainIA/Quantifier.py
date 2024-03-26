import math
import matplotlib
from data_config import *
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Quantifier:
    def __init__(self):
        self.objects = None
        self.start_time = 0

    def prepare_object_list(self, project_directory):
        self.objects = {}
        if project_directory[-1] == "/":
            project_directory = project_directory[0:-1]
        config_file_name = os.path.split(project_directory)[-1] + "_config.json"
        config_file_path = os.path.join(project_directory, config_file_name)
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)
            classes = config_data["classes"]
            for key, value in classes.items():
                if key == "mangeoir":
                    key = "mangeoire"
                self.objects[key] = {"occurrenceNumber": 0, "occurrenceLocation": {}, "avgConfidence": 0}

    def quantify(self, project_directory):
        # Loop through all JSON files in the specified directory
        json_directory = os.path.join(project_directory, "jsonFolder")
        for filename in os.listdir(json_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(json_directory, filename)
                frame_name = filename.split(".json")[0].split("_")[-1]
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    classes_data = json_data[frame_name]["data"]
                    self.start_time = time.time()
                    total_num_classes = len(classes_data)
                    for class_num, class_data in enumerate(classes_data):
                        object_list = classes_data[class_data]
                        occurrence_counter = 0
                        confidence_list = []
                        for obj in object_list:
                            if self.is_number(obj):
                                occurrence_counter += 1
                                # Check if 'conf' or 'confidence' key exists in the object dictionary
                                if 'conf' in object_list[obj]:
                                    confidence_list.append(object_list[obj]['conf'])
                                elif 'confidence' in object_list[obj]:
                                    confidence_list.append(object_list[obj]['confidence'])
                                else:
                                    # Handle the case where neither 'conf' nor 'confidence' key exists
                                    print(f"No confidence information found for object {obj}")
                        # Calculate average confidence
                        if len(confidence_list) != 0:
                            avg_confidence = sum(confidence_list) / len(confidence_list)
                        else:
                            avg_confidence = 0
                        # Update occurrence information
                        self.objects[class_data]["occurrenceNumber"] += 1
                        self.objects[class_data]["occurrenceLocation"][frame_name] = {
                            "occurrenceCounter": occurrence_counter,
                            "confidenceList": confidence_list,
                            "avgConfidence": avg_confidence
                        }
                        # Print progress bar
                        print_progress_bar(self.start_time, class_num + 1, total_num_classes)
                    print_progress_bar(self.start_time, total_num_classes, total_num_classes)


    def plotGraphs(self, type):
        if type == "plotOccurrenceNumber":
            self.plot_occurrence_number()
        if type == "plotSpreadness":
            self.plot_occurrence_spread()
        if type == "plotOccurrenceOverTime":
            self.plot_occurrence_over_time("together")
        if type == "plotConfidenceAverage":
            object_avg_conf = self.analyse_conf_detection()
            self.plot_global_avg_confidence(object_avg_conf)

    def plot_occurrence_number(self):
        object_names = []
        occurrence_numbers = []

        for obj_name, obj_data in self.objects.items():
            if obj_data["occurrenceNumber"] > 0:  # Check if occurrenceCounter is greater than 0
                object_names.append(obj_name)
                occurrence_numbers.append(obj_data["occurrenceNumber"])

        plt.figure(figsize=(10, 6))
        plt.bar(object_names, occurrence_numbers, color='skyblue')
        plt.xlabel('Objects')
        plt.ylabel('Occurrence Number')
        plt.title('Occurrence Number of Objects')
        plt.xticks(rotation=90)
        plt.show()

    def plot_occurrence_spread(self):
        plt.figure(figsize=(10, 6))
        for obj_name, obj_data in self.objects.items():
            frames_dict = obj_data["occurrenceLocation"]  # Assuming "occurrenceLocation" is a list of dictionaries
            frame_numbers = []
            sorted_frames = {key: frames_dict[key] for key in sorted(frames_dict, key=lambda x: int(x.split("frame")[1]))}

            for frame, frame_info in sorted_frames.items():
                if frame_info["occurrenceCounter"] > 0:  # Check if occurrenceCounter is greater than 0
                    # Extract the frame number from the keys of the dictionary
                    frame_number = int(frame.split("frame")[-1])
                    frame_numbers.append(frame_number)

            if frame_numbers:
                plt.plot(frame_numbers, [obj_name] * len(frame_numbers), 'o', label=obj_name)
        plt.xlabel('Frame Number')
        plt.ylabel('Objects')
        plt.title('Spread of Object Detection in Frames')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_occurrence_over_time(self, version):
        if version == "together":
            self.plot_occurrence_over_time_together()
        if version == "split":
            self.plot_occurrence_over_time_split()

    def plot_occurrence_over_time_together(self):
        plt.figure(figsize=(10, 6))
        for obj_name, obj_data in self.objects.items():
            frame_numbers = []
            occurrence_counts = []

            sorted_frames = dict(sorted(obj_data["occurrenceLocation"].items(), key=lambda x: int(x[0].split("frame")[1])))

            # Extract frame numbers and occurrence counts for the current object
            for frame_name in sorted_frames:
                frame_number = int(frame_name.split("frame")[-1])
                occurrence_count = sorted_frames[frame_name]["occurrenceCounter"]
                frame_numbers.append(frame_number)
                occurrence_counts.append(occurrence_count)

            # Check if any occurrence count is greater than zero for the current object
            if any(occurrence_counts):
                # Plotting the occurrence counts over time for the current object
                plt.plot(frame_numbers, occurrence_counts, label=obj_name)

        plt.xlabel('Frame Number')
        plt.ylabel('Occurrence Count')
        plt.title('Occurrence of Objects Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_occurrence_over_time_split(self):
        # Filter objects with occurrences
        objects_with_occurrences = {obj_name: obj_data for obj_name, obj_data in self.objects.items() if any(
            list(frame_dict.values())[0] for frame_dict in obj_data["occurrenceLocation"])}

        num_objects = len(objects_with_occurrences)
        num_rows = math.ceil(math.sqrt(num_objects))
        num_cols = math.ceil(num_objects / num_rows)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows), sharex=True)

        colormap = plt.cm.get_cmap('tab10', num_objects)

        plot_index = 0
        for obj_name, obj_data in objects_with_occurrences.items():
            frame_numbers = []
            occurrence_counts = []

            # Sort the list of dictionaries based on frame numbers
            sorted_frames = sorted(obj_data["occurrenceLocation"],
                                   key=lambda x: int(list(x.keys())[0].split("frame")[-1]))

            # Extract frame numbers and occurrence counts for the current object
            for frame_dict in sorted_frames:
                frame_number = int(list(frame_dict.keys())[0].split("frame")[-1])
                occurrence_count = list(frame_dict.values())[0]
                frame_numbers.append(frame_number)
                occurrence_counts.append(occurrence_count)

            # Plotting the occurrence counts over time for the current object
            row = plot_index // num_cols
            col = plot_index % num_cols
            ax = axes[row, col] if num_objects > 1 else axes

            ax.plot(frame_numbers, occurrence_counts, color=colormap(plot_index))
            ax.set_ylabel('Occurrence Count')
            ax.set_title(obj_name)
            ax.set_xlabel('Frame Number')

            plot_index += 1
        plt.suptitle('Occurrence of Objects Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def analyse_conf_detection(self):
        # Initialize dictionaries to store average confidence for each object and for each frame
        object_average_confidence = {}
        # print("\n")
        for obj_name, obj_data in self.objects.items():
            object_total_confidence_list = []
            total_objects_in_sequence = 0

            # Sort the list of dictionaries based on frame numbers
            sorted_frames = sorted(obj_data["occurrenceLocation"].items(), key=lambda x: int(x[0].split("frame")[-1]))
            for frameName, frame_data in sorted_frames:
                if frame_data["occurrenceCounter"] > 0:
                    for conf in frame_data["confidenceList"]:
                        object_total_confidence_list.append(conf)
                    total_objects_in_sequence += frame_data["occurrenceCounter"]

            if total_objects_in_sequence != len(object_total_confidence_list):
                print("NOT THE SAME AMOUNT OF AVG AND OBJECTS")

            if total_objects_in_sequence > 0:
                object_conf_avg = sum(object_total_confidence_list) / total_objects_in_sequence
                object_average_confidence[obj_name] = object_conf_avg
                # print(obj_name, f"{object_conf_avg:.4f}")

        return object_average_confidence


    def plot_global_avg_confidence(self, object_average):
        plt.figure(figsize=(10, 6))
        plt.bar(object_average.keys(), object_average.values(), color='skyblue')
        plt.xlabel('Object Name')
        plt.ylabel('Average Confidence')
        plt.title('Average Confidence of Objects')
        plt.xticks(rotation=45, ha='right')  # Rotate X labels for better readability
        plt.tight_layout()
        plt.show()

    def is_number(self, s):
        try:
            float(s)  # Try to convert the string to a float
            return True
        except ValueError:
            return False


if __name__ == "__main__":
    quantifier = Quantifier()
    project_directory = f"C:/Users/david/Documents/Brain_Projects/devdavid_laptop_12/frameNumber/20240325_170729"
    quantifier.prepare_object_list(project_directory)
    quantifier.quantify(project_directory)
    quantifier.plot_occurrence_number()
    quantifier.plot_occurrence_spread()
    quantifier.plot_occurrence_over_time("together")

    object_avg_conf = quantifier.analyse_conf_detection()
    quantifier.plot_global_avg_confidence(object_avg_conf)



