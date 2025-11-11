import os
import shutil
import random
from pathlib import Path
import yaml

# ------------------ Settings ------------------
dataset_path = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Dataset"
output_path = r"C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\SplitData"
categories = ["Real", "Fake"]
split_ratio = {"train": 0.7, "val": 0.2, "test": 0.1}

# ------------------ Create folders ------------------
for split in ["train", "val", "test"]:
    for folder_type in ["images", "labels"]:
        os.makedirs(os.path.join(output_path, split, folder_type), exist_ok=True)

# ------------------ Split function ------------------
def split_category(category):
    images_path = Path(dataset_path) / category
    labels_path = Path(dataset_path) / (category + "_labels")
    all_images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    random.shuffle(all_images)

    n = len(all_images)
    train_end = int(split_ratio["train"] * n)
    val_end = train_end + int(split_ratio["val"] * n)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split_name, files in splits.items():
        for img_file in files:
            # Copy image
            dest_img = os.path.join(output_path, split_name, "images", img_file.name)
            shutil.copy(img_file, dest_img)

            # Copy corresponding label
            label_file = labels_path / (img_file.stem + ".txt")
            dest_label = os.path.join(output_path, split_name, "labels", label_file.name)
            if label_file.exists():
                shutil.copy(label_file, dest_label)

# ------------------ Split all categories ------------------
for category in categories:
    split_category(category)

# ------------------ Create data.yaml ------------------
data_yaml = {
    "train": str(Path(output_path) / "train" / "images"),
    "val": str(Path(output_path) / "val" / "images"),
    "test": str(Path(output_path) / "test" / "images"),
    "nc": len(categories),
    "names": categories
}

yaml_path = os.path.join(output_path, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print("Data split completed. 'data.yaml' created for YOLO training.")
