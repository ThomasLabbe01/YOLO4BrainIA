import os
import shutil
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import flickrapi
import urllib
from PIL import Image
import urllib.request
import sys
import random
class_labels_dict = {
    0: 'apple',
    1: 'backpack',
    2: 'banana',
    3: 'bottle',
    4: 'bowl',
    5: 'cell phone',
    6: 'chair',
    7: 'cup',
    8: 'fork',
    9: 'keyboard',
    10: 'knife',
    11: 'laptop',
    12: 'mouse',
    13: 'orange',
    14: 'remote',
    15: 'spoon',
    16: 'suitcase',
    17: 'teddy bear',
    18: 'tie',
    19: 'umbrella',
    20: 'wine glass',
    21: 'human hand',
    22: 'door handle',
    23: 'boot',
    24: 'glove',
    25: 'clothing',
    26: 'sock',
    27: 'book',
    28: 'scissors',
    29: 'wallet',
    30: 'keys',
    31: 'brush',
    32: 'toothbrush',
    33: 'hairdryer',
    34: 'plate',
    35: 'spatula',
    36: 'towel',
    37: 'sponge',
    38: 'pillow',
    39: 'blanket',
    40: 'lamp',
    41: 'pen',
    42: 'notebook',
    43: 'marker',
    44: 'stapler',
    45: 'tape',
    46: 'calculator',
    47: 'headphones',
    48: 'microphone',
    49: 'camera',
    50: 'glasses',
    51: 'hat',
    52: 'scarf',
    53: 'watch',
    54: 'bracelet',
    55: 'ring',
    56: 'belt',
    57: 'shoe',
    58: 'sandal'
}
class_labels_format_open = {10: 0, 15: 1, 21: 2, 40: 56, 54: 27, 56: 23, 57: 3, 60: 4, 80: 46, 82: 49, 104: 6, 115: 25, 165: 22, 204: 8, 217: 50, 218: 24, 243: 51, 244: 47, 267: 21, 296: 10, 301: 40, 304: 11, 331: 48, 343: 12, 356: 13, 376: 41, 388: 38, 395: 34, 432: 58, 437: 52, 438: 28, 476: 26, 480: 35, 483: 15, 490: 44, 503: 16, 525: 17, 533: 18, 542: 32, 545: 36, 562: 19, 577: 53, 590: 20}
class_labels_format_coco = {24: 1, 25: 19, 27: 18, 28: 16, 39: 3, 40: 20, 41: 7, 42: 8, 43: 10, 44: 15, 45: 4, 46: 2, 47: 0, 49: 13, 56: 6, 63: 11, 64: 12, 65: 14, 66: 9, 67: 5, 73: 27, 76: 28, 77: 17, 79: 32}

def flatten_folders(root_dir):
    print("Formatting raw folder of coco and open images from fiftyone")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            source_path = os.path.join(root, file)
            # Extract the filename without the 'val' directory
            dest_path = os.path.join(root_dir, file)
            shutil.move(source_path, dest_path)
    val_dir = os.path.join(root_dir, 'val')
    os.rmdir(val_dir)

def copy_labels(folder):
    print("Creating a mixed label folder")
    folder = os.path.join(folder, "labels")
    dest_folder = folder + '_mixed'
    try:
        # Copy the folder recursively
        shutil.copytree(folder, dest_folder)
    except FileExistsError:
        print(f"Destination folder '{dest_folder}' already exists.")


def relabel(folder):
    # Load the YAML file containing item names
    with open(f'{folder}/dataset.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    
    # Traverse through the label files and count instances
    label_folder = os.path.join(folder, "labels_mixed")
    total_files = len([file_name for file_name in os.listdir(label_folder) if file_name.endswith('.txt')])
    progress_interval = total_files // 20  # Print advancements every 5%

    current_progress = 0
    current_file_count = 0
    
    if 'coco' in folder:
        data_dict = class_labels_format_coco
    else:
        data_dict = class_labels_format_open


    for file_name in os.listdir(label_folder):
        if file_name.endswith('.txt'):
            with open(os.path.join(label_folder, file_name), 'r') as label_file:
                lines = label_file.readlines()
        modified_content=''
        with open(os.path.join(label_folder, file_name), 'w') as label_file:
            for line in lines:
                item_id = int(line.split()[0]) 
                item_name = config['names'][item_id].lower()
                if item_name in class_labels_dict.values():
                    new_item_id = data_dict[item_id]
                    modified_content = line.replace(str(item_id), str(new_item_id), 1)
                    label_file.write(modified_content)

        current_file_count += 1
        if current_file_count >= current_progress + progress_interval:
            print(f"Progress: {round(100*current_file_count/total_files,1)}%")
            current_progress = current_file_count
    
    # rename folder to avoid overwrite
    os.rename(label_folder, label_folder+'_relabeled')

def find_filename(filename):
    while os.path.exists(filename):
        filename = filename.split('.')[0]
        filename = filename[:-1] + str(int(filename[-1])+1) + '.txt'
    return filename

def sample(dataset_folder, filter):
    print("Creating txt files in respective directory. sampling the classes and getting a more uniform distribution of instances")
    # Load the YAML file containing item names
    dataset_sample = []
    with open(f'{dataset_folder}/dataset.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Initialize a dictionary to store item counts
    item_counts = {item_name: 0 for item_name in class_labels_dict.keys()}
    # Traverse through the label files and count instances
    label_folder = f'{dataset_folder}/labels_mixed_relabeled'
    total_files = len([file_name for file_name in os.listdir(label_folder) if file_name.endswith('.txt')])
    progress_interval = total_files // 20  # Print advancements every 5%

    current_progress = 0
    current_file_count = 0

    for file_name in os.listdir(label_folder):
        valid_file = []
        if file_name.endswith('.txt'):
            with open(os.path.join(label_folder, file_name), 'r') as label_file:
                for line in label_file:
                    item_id = int(line.split()[0])
                    if item_counts[item_id] > filter:
                        valid_file.append(False)
                    else:
                        valid_file.append(True)
                if True in valid_file:
                    with open(os.path.join(label_folder, file_name), 'r') as label_file:
                        for line in label_file:
                            item_id = int(line.split()[0])
                            item_counts[item_id] += 1
                        dataset_sample.append(file_name)
                current_file_count += 1
            if current_file_count >= current_progress + progress_interval:
                print(f"Progress: {round(100*current_file_count/total_files,1)} %")
                current_progress = current_file_count
    
    
    # Save selected images in a text file
    dataset_name = dataset_folder.split('\\')[-1]
    filename = dataset_folder + '/' + dataset_name + "-sample0.txt"
    filename = find_filename(filename)
    with open(filename, "w") as file:
        for item in dataset_sample:
            file.write(item + "\n")  # Add a newline after each item
                
    return dataset_sample, item_counts

def get_bg(amount):
    print("Getting the background images from flickr. all image start with 'bg'")
    # Flickr api access key 
    flickr=flickrapi.FlickrAPI('36a99772268bd98801231e28b0f8d56a', 'f94f8efd54cb358e', cache=True)

    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "background_images")):
        sys.exit("Background images are already loaded. Delete the folder to download new images, or set background_images_bool==False ")
    os.mkdir(os.path.join(cwd, "background_images"))
    os.mkdir(os.path.join(cwd, "background_images", 'images'))
    os.mkdir(os.path.join(cwd, "background_images", 'labels'))
    os.mkdir(os.path.join(cwd, "background_images", 'labels_mixed_relabeled'))


    keyword_list = ['siberian husky', "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "bench", "cat", "dog", "horse", "sheep", "bear", "elephant", "cow", "stop sign", "fire hydrant", "kite", "landscape", "tree", "forest", "rocks", "ocean", "fish", "plant", "building", "parking lot", "flags", "elevator", "floor", "road", "bush", "mine", "forest"]

    for keyword in keyword_list:
        photos = flickr.walk(text=keyword,
                        tag_mode='all',
                        tags=keyword,
                        extras='url_c',
                        per_page=50,           
                        sort='relevance')

        urls = []
        images_name = []
        for i, photo in enumerate(photos):
            url = photo.get('url_c')
            if url == None:
                continue
            name = url.split('/')[4][:-6]
            urls.append(url)
            images_name.append(name)
            # number of image for every item in list
            if i > amount/len(keyword_list):
                break
        # Download image from the url 
        for i in range(len(urls)):
            id = f'background_{images_name[i]}'
            urllib.request.urlretrieve(urls[i], id + ".jpg")
                # Resize the image and overwrite it
            image = Image.open(id + ".jpg") 
            image = image.resize((640, 480))
            image.save(os.path.join(cwd, "background_images", "images",id + ".jpg"))
            open(os.path.join(cwd, "background_images", "labels", id + ".txt"), mode='a').close()
            open(os.path.join(cwd, "background_images", "labels_mixed_relabeled", id + ".txt"), mode='a').close()
            os.remove(id + ".jpg")

def get_new_objects(amount):
    print("Getting extra images from flickr. all image start with 'xtra'")
    # Flickr api access key 
    flickr=flickrapi.FlickrAPI('36a99772268bd98801231e28b0f8d56a', 'f94f8efd54cb358e', cache=True)

    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "flickr_extra_class")):
        sys.exit("flickr_extra_class images are already loaded. Delete the folder to download new images, or set background_images_bool==False ")
    os.mkdir(os.path.join(cwd, "flickr_extra_class"))
    os.mkdir(os.path.join(cwd, "flickr_extra_class", 'images'))
    os.mkdir(os.path.join(cwd, "flickr_extra_class", 'labels'))
    os.mkdir(os.path.join(cwd, "flickr_extra_class", 'labels_mixed_relabeled'))


    keyword_list = ["wallet", "keys", "brush", "hair dryer", "sponge", "blanket", "notebook", "marker", "tape", "bracelet", "ring", "shoe"]

    for keyword in keyword_list:
        photos = flickr.walk(text=keyword,
                        tag_mode='all',
                        tags=keyword,
                        extras='url_c',
                        per_page=50,           
                        sort='relevance')

        urls = []
        images_name = []
        for i, photo in enumerate(photos):
            url = photo.get('url_c')
            if url == None:
                continue
            name = url.split('/')[4][:-6]
            urls.append(url)
            images_name.append(keyword + '_' + name)
            # number of image for every item in list
            if i > amount:
                break
        # Download image from the url 
        for i in range(len(urls)):
            id = f'xtra_{images_name[i]}'
            urllib.request.urlretrieve(urls[i], id + ".jpg")
                # Resize the image and overwrite it
            image = Image.open(id + ".jpg") 
            image = image.resize((640, 480))
            image.save(os.path.join(cwd, "flickr_extra_class", "images",id + ".jpg"))
            open(os.path.join(cwd, "flickr_extra_class", "labels", id + ".txt"), mode='a').close()
            open(os.path.join(cwd, "flickr_extra_class", "labels_mixed_relabeled", id + ".txt"), mode='a').close()
            os.remove(id + ".jpg")

def move_images(source_path, dest_path, train_val_split):
    image_path = os.path.join(source_path, "images")
    label_path = os.path.join(source_path, "labels")
    label_mixed_relabeled_path = os.path.join(source_path, "labels_mixed_relabeled")
    files = os.listdir(image_path)
    random.seed(42)
    random.shuffle(files)
    files = [file_name[:-4] for file_name in files]
    total_files = len(files)
    num_files_train = int(total_files * bg_train_split)
    files_train = files[:num_files_train]
    files_validation = files[num_files_train:]

    for file in files_train:
        source_image = os.path.join(image_path, file + ".jpg")
        destination_image = os.path.join(dest_path, "train", 'images', file + ".jpg")
        if not os.path.exists(destination_image):
            shutil.copy(source_image, destination_image)

        source_label = os.path.join(label_path, file + ".txt")
        destination_label = os.path.join(dest_path, "train", 'labels', file + ".txt")
        if not os.path.exists(destination_label):
            shutil.copy(source_label, destination_label)

        source_label_mixed = os.path.join(label_mixed_relabeled_path, file + ".txt")
        destination_label_mixed = os.path.join(dest_path, "train", "labels_mixed_relabeled", file + ".txt")
        if not os.path.exists(destination_label_mixed):
            shutil.copy(source_label_mixed, destination_label_mixed)

    for file in files_validation:
        source_image = os.path.join(image_path, file + ".jpg")
        destination_image = os.path.join(dest_path, "validation", 'images', file + ".jpg")
        if not os.path.exists(destination_image):
            shutil.copy(source_image, destination_image)

        source_label = os.path.join(label_path, file + ".txt")
        destination_label = os.path.join(dest_path, "validation", 'labels', file + ".txt")
        if not os.path.exists(destination_label):
            shutil.copy(source_label, destination_label)

        source_label_mixed = os.path.join(label_mixed_relabeled_path, file + ".txt")
        destination_label_mixed = os.path.join(dest_path, "validation", "labels_mixed_relabeled", file + ".txt")
        if not os.path.exists(destination_label_mixed):
            shutil.copy(source_label_mixed, destination_label_mixed)
        
def create_final_dataset_folder(dataset_folders, bg_path, flickr_path, train_val_split, source_folder):
    print("Making the combined dataset folder")
    '''
    3 config.yaml
    images des 4 datasets fitrés par lex txt files et répertorié avec train et validation
        train -- images - labels - labels_mixed_relabeled
        validation -- images - labels - labels_mixed_relabeled
        inclure dans train et validation 70-30 background
        inclure image flickr 70-30 
    '''
    # Create the dataset folder
    combined_dataset_folder = os.path.join(source_folder, "Combined_dataset")
    if not os.path.exists(combined_dataset_folder):
        os.makedirs(combined_dataset_folder)
    
    # Get .yaml files from coco-train and open-images-train
    yaml_directory = [dataset_folders[0], dataset_folders[2]]
    for path in yaml_directory:
        yaml_name = path.split('\\')[-1]
        source_file = os.path.join(path, "dataset.yaml")
        destination_file  = os.path.join(combined_dataset_folder, yaml_name + '-config.yaml')
        if os.path.exists(source_file):
            shutil.move(source_file, destination_file)

    # Move combined .yaml file to combined_dataset
    source_file = os.path.join(source_folder, "Important", "combined_config.yaml")
    destination_file  = os.path.join(combined_dataset_folder, 'combined_config.yaml')
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)

    # Create training repo correctly
    if not os.path.exists(os.path.join(combined_dataset_folder, 'train')):
        os.mkdir(os.path.join(combined_dataset_folder, 'train')) 
        os.mkdir(os.path.join(combined_dataset_folder, 'train', 'images')) 
        os.mkdir(os.path.join(combined_dataset_folder, 'train', 'labels')) 
        os.mkdir(os.path.join(combined_dataset_folder, 'train', 'labels_mixed_relabeled')) 
    if not os.path.exists(os.path.join(combined_dataset_folder, 'validation')):
        os.mkdir(os.path.join(combined_dataset_folder, 'validation')) 
        os.mkdir(os.path.join(combined_dataset_folder, 'validation', 'images')) 
        os.mkdir(os.path.join(combined_dataset_folder, 'validation', 'labels')) 
        os.mkdir(os.path.join(combined_dataset_folder, 'validation', 'labels_mixed_relabeled')) 

    # Move images
    file_to_sample_list = [[], [], [], []]
    c = -1
    for dataset_folder in dataset_folders:
        c += 1
        for file in os.listdir(dataset_folder):
            if "sample0" in os.path.basename(file) and os.path.basename(file).endswith('.txt'):
                with open(os.path.join(dataset_folder, file), 'r') as sample_indications:
                    for file_to_sample in sample_indications:
                        file_to_sample_list[c].append(file_to_sample[:-5])

    c=-1
    split = ['train', 'validation', 'train', 'validation']
    print('Moving sampled coco and open images to combined dataset')   
    for dataset_folder in dataset_folders:
        c+=1
        for file in file_to_sample_list[c]:

            source_image = os.path.join(dataset_folder, "images", file + ".jpg")
            destination_image = os.path.join(combined_dataset_folder, split[c], 'images', file + ".jpg")
            if not os.path.exists(destination_image):
                shutil.copy(source_image, destination_image)

            source_label = os.path.join(dataset_folder, "labels", file + ".txt")
            destination_label = os.path.join(combined_dataset_folder, split[c], 'labels', file + ".txt")
            if not os.path.exists(destination_label):
                shutil.copy(source_label, destination_label)

            source_label_mixed = os.path.join(dataset_folder, "labels_mixed_relabeled", file + ".txt")
            destination_label_mixed = os.path.join(combined_dataset_folder, split[c], "labels_mixed_relabeled", file + ".txt")
            if not os.path.exists(destination_label_mixed):
                shutil.copy(source_label_mixed, destination_label_mixed)

    if os.path.exists(bg_path):
        print('Moving background images to combined dataset')   
        move_images(source_path=bg_path, dest_path=combined_dataset_folder, train_val_split=train_val_split)
    
    if os.path.exists(flickr_path):   
        print("Moving flickr images to combined dataset")  
        move_images(source_path=flickr_path, dest_path=combined_dataset_folder, train_val_split=train_val_split)
    
    # Move 
    combined_yaml_file = os.path.join(source_folder, "combined_config.yaml")
    if os.path.exists(combined_yaml_file):
        shutil.copy(combined_yaml_file, combined_dataset_folder)


def count_instances(folder):
    # Load the YAML file containing item names
    with open(f'{folder}/combined_config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Initialize a dictionary to store item counts
    item_counts = {item_key: 0 for item_key in class_labels_dict.keys()}
    
    # Traverse through the label files and count instances
    label_folder = f'{folder}/labels/val'
    total_files = len([file_name for file_name in os.listdir(label_folder) if file_name.endswith('.txt')])
    progress_interval = total_files // 100  # Print advancements every 5%

    current_progress = 0
    current_file_count = 0
    
    for file_name in os.listdir(label_folder):
        if file_name.endswith('.txt'):
            with open(os.path.join(label_folder, file_name), 'r') as label_file:
                for line in label_file:
                    item_id = int(line.split()[0])
                    if item_id in class_labels_dict.keys():
                        item_counts[item_id] += 1
            current_file_count += 1
            if current_file_count >= current_progress + progress_interval:
                print(f"Progress: {100*current_file_count/total_files}")
                current_progress = current_file_count

def sort_dict_by_instances(dictionary):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}

def merge_dicts(*dicts):
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
    return merged_dict

def histogramme_train_validation(combined_dataset_path):
    with open(f'{combined_dataset_path}/combined_config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    # Initialize a dictionary to store item counts
    item_counts_train = {item_id: 0 for item_id in class_labels_dict.keys()}
    item_counts_val = {item_id: 0 for item_id in class_labels_dict.keys()}
    # Traverse through the label files and count instances
    label_folders = [f'{combined_dataset_path}/train/labels_mixed_relabeled/',
                     f'{combined_dataset_path}/validation/labels_mixed_relabeled/',]
    
    item_counts_split = [item_counts_train, item_counts_val]
    i = 0
    for folder in label_folders:
        for file_name in os.listdir(folder):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder, file_name), 'r') as label_file:
                    for line in label_file:
                        item_id = int(line.split()[0])
                        if item_id in class_labels_dict.keys():
                            item_counts_split[i][item_id] += 1
        i+=1

    categories = list(item_counts_train.keys())
    item_counts_hist = {}
    data_to_plot = [{}, {}]
    i=0
    for item_count in item_counts_split:
        for key, value in class_labels_dict.items():
            item_counts_hist[value] = item_count[key]
        data_to_plot[i] = item_counts_hist
        i +=1
        item_counts_hist = {}
    data_to_plot_mixed = merge_dicts(data_to_plot[0], data_to_plot[1])
    data_to_plot_mixed_sorted = sort_dict_by_instances(data_to_plot_mixed)
    sort = True
    if sort:
        train_data = {k: data_to_plot[0][k] for k in data_to_plot_mixed_sorted.keys()}
        val_data = {k: data_to_plot[1][k] for k in data_to_plot_mixed_sorted.keys()}

    train_data = list(train_data.values())
    val_data = list(val_data.values())


    width = 0.9  # Width of the bars
    frontsize = 22
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(top=0.94,
    bottom=0.19,
    left=0.045,
    right=0.99,
    hspace=0.2,
    wspace=0.2)

    ax.set_xlabel('Object Classes', fontsize=frontsize)
    ax.set_title('Instances per category', fontsize=frontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Define a custom formatter function
    def format_y_ticks(value, pos):
        if value < 1000:
            return f"{value:.0f}"
        elif value < 10000:
            return f"{value/1000:.0f}k"
        else:
            return f"{value/1000:.0f}k"

    # Set the custom formatter for y-axis ticks
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

    plt.bar(data_to_plot_mixed_sorted.values(), train_data, label='train data', align='center', width=0.8)
    plt.bar(data_to_plot_mixed_sorted.values(), val_data, bottom=train_data, label='validation data', align='center', width=0.8)    # Set x-axis tick labels rotation
    
    plt.xticks(rotation=45, ha="right", rotation_mode='anchor')
    # Set y-axis to logarithmic scale
    #plt.yscale('log')
    # Add a horizontal line at y=10000
    plt.axhline(y=10000, color='black', linestyle='--', label='Recommendation')

    plt.xlabel('Categories')
    plt.ylabel('Instances')
    plt.title('Dataset presentation')
    plt.legend()
    ax.legend(fontsize=16)
    plt.show()

def histogram_dataset(combined_dataset_path):
    with open(f'{combined_dataset_path}/combined_config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    # Initialize a dictionary to store item counts
    item_counts_train = {item_id: 0 for item_id in class_labels_dict.keys()}
    item_counts_train["-1"] = 0 # background
    item_counts_val = {item_id: 0 for item_id in class_labels_dict.keys()}
    item_counts_val["-1"] = 0 # background
    flickr_association = {"wallet" : 29, "keys" : 30, "brush" : 31, "hair dryer" : 33, "sponge" : 37, "blanket" : 39,"notebook" : 42,"marker" : 43,"tape" : 45,"bracelet": 54, "ring": 55,"shoe": 57}
    # Traverse through the label files and count instances
    label_folders = [f'{combined_dataset_path}/train/labels_mixed_relabeled/',
                     f'{combined_dataset_path}/validation/labels_mixed_relabeled/',]
     
    item_counts_split = [item_counts_train, item_counts_val]
    i = 0
    for folder in label_folders:
        for file_name in os.listdir(folder):
            if file_name.endswith('.txt'):
                for item in flickr_association:
                    if "background" in file_name:
                        item_counts_split[i]["-1"] += 1
                        break
                    if item in file_name:
                        item_counts_split[i][flickr_association[item]] += 1
                        break
                with open(os.path.join(folder, file_name), 'r') as label_file:
                    for line in label_file:
                        item_id = int(line.split()[0])
                        if item_id in class_labels_dict.keys():
                            item_counts_split[i][item_id] += 1
        i+=1
    item_counts_hist = {}
    data_to_plot = [{}, {}]
    i=0
    for item_count in item_counts_split:
        for key, value in class_labels_dict.items():
            item_counts_hist[value] = item_count[key]
        item_counts_hist["background"] = item_count['-1']
        data_to_plot[i] = item_counts_hist
        i +=1
        item_counts_hist = {}
    data_to_plot_mixed = merge_dicts(data_to_plot[0], data_to_plot[1])
    data_to_plot_mixed_sorted = sort_dict_by_instances(data_to_plot_mixed)
    sort = True
    if sort:
        train_data = {k: data_to_plot[0][k] for k in data_to_plot_mixed_sorted.keys()}
        val_data = {k: data_to_plot[1][k] for k in data_to_plot_mixed_sorted.keys()}

    train_data = list(train_data.values())
    val_data = list(val_data.values())


    width = 0.9  # Width of the bars
    frontsize = 22
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(top=0.94,
    bottom=0.19,
    left=0.045,
    right=0.99,
    hspace=0.2,
    wspace=0.2)

    ax.set_xlabel('Object Classes', fontsize=frontsize)
    ax.set_title('Instances per category', fontsize=frontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Define a custom formatter function
    def format_y_ticks(value, pos):
        if value < 1000:
            return f"{value:.0f}"
        elif value < 10000:
            return f"{value/1000:.0f}k"
        else:
            return f"{value/1000:.0f}k"

    # Set the custom formatter for y-axis ticks
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))


    plt.bar(data_to_plot_mixed_sorted.keys(), train_data, label='train data', align='center', width=0.8)
    plt.bar(data_to_plot_mixed_sorted.keys(), val_data, bottom=train_data, label='validation data', align='center', width=0.8)    # Set x-axis tick labels rotation
    
    plt.xticks(rotation=45, ha="right", rotation_mode='anchor')
    # Set y-axis to logarithmic scale
    #plt.yscale('log')
    # Add a horizontal line at y=10000
    plt.axhline(y=10000, color='black', linestyle='--', label='Recommendation')

    plt.xlabel('Categories')
    plt.ylabel('Instances')
    plt.title('Dataset presentation')
    plt.legend()
    ax.legend(fontsize=16)
    plt.show()

if __name__ == "__main__":
    cwd = os.getcwd()

    dataset_folders = [os.path.join(cwd, "coco-train"),
                       os.path.join(cwd, "coco-val"),
                       os.path.join(cwd, "open-images-train"),
                       os.path.join(cwd, "open-images-val")]
    
    # Step 0 : Run DowloadDataset.py
    
    # Step 1 : format initial folder
    for dataset_folder in dataset_folders:
        continue
        print(f"Step 1 : {dataset_folder}")
        flatten_folders(os.path.join(dataset_folder, "images"))
        flatten_folders(os.path.join(dataset_folder, "labels"))

    # Step 2 : copy labels
    for dataset_folder in dataset_folders:
        continue
        print(f"Step 2 : {dataset_folder}")
        copy_labels(dataset_folder)

    # Step 3 : create new labels
    for dataset_folder in dataset_folders:
        continue
        print(f"Step 3 : {dataset_folder}")
        relabel(dataset_folder)
    
    # Step 4 : sample dataset
    for dataset_folder in dataset_folders:
        continue
        sample(dataset_folder, 1000)[1]
        
    # Step 5 : get background images
    background_images_bool = False
    if background_images_bool:
        get_bg(amount=250) # total amount of background

    # Step 6 : get objects that are not from coco or openimages 
    new_objects_bool = False
    if new_objects_bool:
        get_new_objects(amount=250) # amount by item

    # Step 7 : move to final folder
    bg_path = os.path.join(cwd, "background_images")
    flickr_path = os.path.join(cwd, "flickr_extra_class")
    bg_train_split = 0.7
    move_final_folder = False
    if move_final_folder:
        create_final_dataset_folder(dataset_folders, bg_path, flickr_path, bg_train_split, cwd)
       
    # Step 8: check histogramme_train_validation
    '''NOT DONE YET'''
    plot_histogram = False
    combined_dataset_path = os.path.join(cwd, 'Combined_dataset')
    if plot_histogram:
        histogramme_train_validation(combined_dataset_path)

    # Step 9 : check histogramme_current_version
    plot_histogram_test = True
    if plot_histogram_test:
        histogram_dataset(combined_dataset_path)

    # Step 10 : change paths in .yaml files


    
