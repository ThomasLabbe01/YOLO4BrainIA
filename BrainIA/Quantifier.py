import os
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data_config import *


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
                        self.objects[classe]["occurenceNumber"] = self.objects[classe]["occurenceNumber"] + 1
                        self.objects[classe]["occurenceLocation"].append(frame_name)
                        print_progress_bar(self.start_time, classe_num, total_num_classe)
                    print_progress_bar(self.start_time, total_num_classe, total_num_classe)


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
            frames = obj_data["occurenceLocation"]
            frame_numbers = [int(frame.split("frame")[-1]) for frame in frames]
            plt.plot(frame_numbers, [obj_name]*len(frame_numbers), 'o', label=obj_name)
        plt.xlabel('Frame Number')
        plt.ylabel('Objects')
        plt.title('Spread of Object Detection in Frames')
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    quantifier = Quantifier()
    project_directory = f"C:/Users/david/Documents/Brain_Projects/devdavid_3/ClasseOrdered/img_3343"
    quantifier.prepare_object_list(project_directory)
    quantifier.quantify(project_directory)
    quantifier.plot_occurrence_number()
    quantifier.plot_occurrence_spread()