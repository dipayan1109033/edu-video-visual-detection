import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils import data
from torchvision import transforms
import albumentations as A

from src.utils.common import Helper
helper = Helper()


# Collate image-target pairs into a tuple.
def collate_fn(batch):
    return tuple(zip(*batch))




# A.SmallestMaxSize(max_size=640, p=1.0)
# A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=0)
# A.RandomScale(scale_limit=0.1, p=1.0)
# A.Rotate(limit=30, p=1.0)

def get_augmentation_pipeline():

    augment_pipeline = A.Compose([
            A.OneOf([
                        A.RandomBrightnessContrast(p=0.5),
                        A.HueSaturationValue(p=0.25),
                    ], p=0.25),
            A.OneOf([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.25),
                    ], p=0.5),
            A.OneOf([
                        A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR, p=0.25),
                        A.ShiftScaleRotate(rotate_limit=15, p=0.5),
                        A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.5),
                    ], p=0.5),
            A.XYMasking(num_masks_x=(1,2), num_masks_y=(1,2), mask_x_length=(10, 50), mask_y_length=(10,50), p=0.15)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=100, min_visibility=0.1, label_fields=['labels']),
        p=0.90)

    return augment_pipeline


# Reading and converting image labels
def json_to_labeldict(json_file_path):
    # Load the JSON data from the file
    with open(json_file_path) as f:
        data = json.load(f)

    image_id = data["asset"]["image_id"]
    label_list, box_list = [], []
    for object in data["objects"]:
        label_list.append(object["class"])
        box_list.append([
            object["boundingBox"]["xmin"], 
            object["boundingBox"]["ymin"], 
            object["boundingBox"]["xmax"], 
            object["boundingBox"]["ymax"]
            ])
    
    label_dict = {"image_id": torch.as_tensor(image_id, dtype=torch.int32),  
                  "boxes": box_list, "labels": label_list}
    return label_dict

# PyTorch Dataset
class LectureFrameODDataset(data.Dataset):
    def __init__(self, dataset_path, transformFlag = False):
        """
        Dataset for object detection tasks.

        Args:
            dataset_path (str): Directory path to the dataset.
            transform_flag (bool, optional): Whether to apply additional image transformation. Defaults to False.
        """
        self.root = dataset_path
        self.augmentFlag = transformFlag
        self.augmentation = get_augmentation_pipeline()
        self.image_files = helper.get_image_files(os.path.join(self.root, "images"))
        self.dataset_size = len(self.image_files)


    def __getitem__(self, i):
        # Load image from the hard disc.
        image_filename = self.image_files[i]
        image_path = os.path.join(self.root, "images", image_filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Load labels and bounding boxes     
        label_dict = json_to_labeldict(os.path.join(self.root, "custom_labels", image_filename[:-4] + ".json"))
        
        # Apply augmentations
        if self.augmentFlag:
            augmented = self.augmentation(image=image, bboxes=label_dict["boxes"], labels=label_dict["labels"])
            image, label_dict["boxes"], label_dict["labels"] = augmented["image"], augmented['bboxes'], augmented['labels']

        if len(label_dict["labels"]) > 0:
            label_dict["boxes"] = torch.as_tensor(label_dict["boxes"], dtype=torch.float32)
            label_dict["labels"] = torch.as_tensor(label_dict["labels"], dtype=torch.int64)
        else:
            label_dict["boxes"] = torch.empty((0, 4), dtype=torch.float32)  # Shape [0, 4] for no boxes
            label_dict["labels"] = torch.tensor([], dtype=torch.int64)

        image = transforms.ToTensor()(image)
        return image, label_dict
    
    def __len__(self):
        return self.dataset_size





def plot_sample_images_from_dataset(torch_dataset, num_images = 2):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # Ensure axes is iterable if num_images is 1
    if num_images == 1:
        axes = [axes]

    # Randomly select indices for images from the dataset
    indices = np.random.choice(len(torch_dataset), num_images, replace=False)

    for i, idx in enumerate(indices):
        # Get the image and labels
        image, target = torch_dataset[idx]

        # Convert the image to a numpy array if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Convert CxHxW to HxWxC format

        # Display the image
        axes[i].imshow(image)
        axes[i].set_xticks([])  # Remove x-ticks
        axes[i].set_yticks([])  # Remove y-ticks

        # Get and plot bounding boxes and labels
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        padding = image.shape[0]*0.01   # 1% of image height
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin

            # Create and add a rectangle patch
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            
            # Add the label text
            axes[i].text(xmin+padding, ymin+padding*4, f'{label}', color='r', fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    # Required parameters
    temp_dataset_dir = "experiments/input/temp_datasets/testdata_1k_80_20_0_seed42"

    train_folder_path = os.path.join(temp_dataset_dir, "train")
    dataset = LectureFrameODDataset(train_folder_path, transformFlag=False)
    print(f"Total samples in dataset: {len(dataset)}")

    plot_sample_images_from_dataset(dataset, 2)


    # Get a sample item
    image, label_dict = dataset[0]
    print(f"Image shape: {image.shape}, Image type: {image.dtype}")
    print(f"Boxes: {label_dict['boxes']}, Labels: {label_dict['labels']}")


if __name__ == "__main__":
    print("In src/data/dataset.py main()")
    main()



