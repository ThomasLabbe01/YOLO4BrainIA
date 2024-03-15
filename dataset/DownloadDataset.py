import fiftyone as fo
import fiftyone.zoo as foz
"https://docs.voxel51.com/tutorials/yolov8.html"
'https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html'
def export_yolo_data(
    samples,
    export_dir,
    classes,
    label_field = "ground_truth",
    split = None
    ):

    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples,
                export_dir,
                classes,
                label_field,
                split
            )
    else:
        if split is None:
            split_view = samples
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split
        )

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes = ['Apple', 'Backpack', 'Banana', 'Bottle', 'Bowl', 'Chair', 'Fork', 'Knife', 'Laptop', 'Mouse', 'Orange', 'Spoon', 'Suitcase', 'Teddy bear', 'Tie', 'Umbrella', 'Wine glass', 'Human hand', 'Door handle', 'Boot', 'Glove', 'Clothing', 'Sock', 'Book', 'Scissors', 'Toothbrush', 'Plate', 'Spatula', 'Towel', 'Pillow', 'Lamp', 'Pen', 'Stapler', 'Calculator', 'Headphones', 'Microphone', 'Camera', 'Glasses', 'Hat', 'Scarf', 'Watch', 'Belt', 'Sandal'],
    # max_samples=5000
)
classes = [c for c in dataset.default_classes if not c.isnumeric()]
dir = "open-images-train"
export_yolo_data(dataset, dir, classes)

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes = ['Apple', 'Backpack', 'Banana', 'Bottle', 'Bowl', 'Chair', 'Fork', 'Knife', 'Laptop', 'Mouse', 'Orange', 'Spoon', 'Suitcase', 'Teddy bear', 'Tie', 'Umbrella', 'Wine glass', 'Human hand', 'Door handle', 'Boot', 'Glove', 'Clothing', 'Sock', 'Book', 'Scissors', 'Toothbrush', 'Plate', 'Spatula', 'Towel', 'Pillow', 'Lamp', 'Pen', 'Stapler', 'Calculator', 'Headphones', 'Microphone', 'Camera', 'Glasses', 'Hat', 'Scarf', 'Watch', 'Belt', 'Sandal'],
    # max_samples=5000
)
classes = [c for c in dataset.default_classes if not c.isnumeric()]
dir = "open-images-val"
export_yolo_data(dataset, dir, classes)

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],   
    classes=['apple', 'backpack', 'banana', 'bottle', 'bowl', 'cell phone', 'chair', 'cup', 'fork', 'keyboard', 'knife', 'laptop', 'mouse', 'orange', 'remote', 'spoon', 'suitcase', 'teddy bear', 'tie', 'umbrella', 'wine glass', 'book', 'scissors', 'toothbrush'],
    # max_samples=5000
)
classes = [c for c in dataset.default_classes if not c.isnumeric()]
dir = "coco-train"
export_yolo_data(dataset, dir, classes)

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=['apple', 'backpack', 'banana', 'bottle', 'bowl', 'cell phone', 'chair', 'cup', 'fork', 'keyboard', 'knife', 'laptop', 'mouse', 'orange', 'remote', 'spoon', 'suitcase', 'teddy bear', 'tie', 'umbrella', 'wine glass', 'book', 'scissors', 'toothbrush'],
    # max_samples=5000
)
classes = [c for c in dataset.default_classes if not c.isnumeric()]
dir = "coco-val"
export_yolo_data(dataset, dir, classes)
