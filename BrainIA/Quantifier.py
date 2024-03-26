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
                self.objects[key] = {"occurenceNumber": 0, "occurenceLocation": []}

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
                    total_num_classe = len(classes_data)
                    for classe_num, classe in enumerate(classes_data):
                        if len(classes_data[classe]) == 0:
                            object_occurence_in_frame = 0
                        else:
                            object_occurence_in_frame = len(classes_data[classe]) - 1  # We have 1 more class of unconfirmed
                        #NEEDS TO BE DONE AFTER THE PREVIOUS LINE
                        if classe == "mangeoire":
                            classe = "mangeoir"
                        self.objects[classe]["occurenceNumber"] += 1
                        self.objects[classe]["occurenceLocation"].append({frame_name: object_occurence_in_frame})
                        print_progress_bar(self.start_time, classe_num, total_num_classe)
                    print_progress_bar(self.start_time, total_num_classe, total_num_classe)

    def plotGraphs(self, type):
        if type == "plotOccurenceNumber":
            self.plot_occurrence_number()
        if type == "plotSpreadness":
            self.plot_occurrence_spread()
        if type == "plotOccurenceOverTime":
            self.plot_occurrence_over_time("together")

    def plot_occurrence_number(self):
        object_names = list(self.objects.keys())
        occurrence_numbers = [obj_data["occurenceNumber"] for obj_data in self.objects.values()]

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
            frames_dict = obj_data["occurenceLocation"]  # Assuming "occurenceLocation" is a list of dictionaries
            frame_numbers = []
            for frame in frames_dict:
                # Extract the frame number from the keys of the dictionary
                frame_number = int(list(frame.keys())[0].split("frame")[-1])
                frame_numbers.append(frame_number)

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

            # Sort the list of dictionaries based on frame numbers
            sorted_frames = sorted(obj_data["occurenceLocation"],
                                   key=lambda x: int(list(x.keys())[0].split("frame")[-1]))

            # Extract frame numbers and occurrence counts for the current object
            for frame_dict in sorted_frames:
                frame_number = int(list(frame_dict.keys())[0].split("frame")[-1])
                occurrence_count = list(frame_dict.values())[0]
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
            list(frame_dict.values())[0] for frame_dict in obj_data["occurenceLocation"])}

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
            sorted_frames = sorted(obj_data["occurenceLocation"],
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


if __name__ == "__main__":
    quantifier = Quantifier()
    # project_directory = f"C:/Users/david/Documents/Brain_Projects/devdavid_3/testingClass/20mice3"
    project_directory = f"C:/Users/david/Documents/Brain_Projects/devdavid_3/ClasseOrdered/img_3343"
    quantifier.prepare_object_list(project_directory)
    quantifier.quantify(project_directory)
    quantifier.plot_occurrence_number()
    quantifier.plot_occurrence_spread()
    quantifier.plot_occurrence_over_time()

