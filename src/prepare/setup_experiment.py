
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import random
import random
from sklearn.model_selection import KFold
from src.prepare.dataset_utils import *
from src.utils.common import Helper



def create_experiment_metadata(root_dir, datasets, metadata_folder, custom_split_code, seed=42):
    """Generates metadata for dataset partitions and saves it as a JSON file.
    
    Args:
        root_dir (str): Root directory containing dataset folders.
        datasets (dict): Dictionary containing dataset configurations.
        metadata_folder (str): Directory to save the metadata file.
        custom_split_code (str): Custom split code for referencing meta filenames.
        seed (int): Random seed used for reproducibility.

    Returns:
        str: Path to the generated metadata JSON file.
    """
    random.seed(seed)
    metadata = {
        "destination_folder": custom_split_code,
        "datasets": datasets,
        "seed": seed,
        "split_data": {}
    }
    
    total_images = {"train": 0, "val": 0, "test": 0}
    
    for dataset_name, dataset_info in datasets.items():
        dataset_path = os.path.join(root_dir, dataset_name, "images")
        image_files = helper.get_image_files(dataset_path)
        
        # Apply percentage filter if less than 100%
        percentage = dataset_info["percentage"]
        if percentage < 100:
            selected_count = int(len(image_files) * (percentage / 100))
            image_files = random.sample(image_files, selected_count)
        
        split_ratio = dataset_info["split_ratio"]
        partitions = helper.split_dataset(image_files, split_ratio)
        
        metadata["split_data"][dataset_name] = {
            "train": partitions["train"],
            "val": partitions["val"],
            "test": partitions["test"]
        }
        
        # Count total images for final split ratio calculation
        total_images["train"] += len(partitions["train"])
        total_images["val"] += len(partitions["val"])
        total_images["test"] += len(partitions["test"])
    
    total = sum(total_images.values())
    if total > 0:
        metadata["final_split_count"] = total_images
        metadata["final_split_ratio"] = {
            "train": round(total_images["train"] / total, 2),
            "val": round(total_images["val"] / total, 2),
            "test": round(total_images["test"] / total, 2)
        }
        
    # Save to JSON file
    os.makedirs(metadata_folder, exist_ok=True)
    metadata_file = os.path.join(metadata_folder, f"{custom_split_code}.json")
    helper.write_to_json(metadata, metadata_file)

    return metadata_file

def create_crossval_experiment_metadata(root_dir, datasets, metadata_folder, custom_split_code, num_folds=5, seed=42):
    """Generates metadata for cross-validation dataset partitions and saves it as a JSON file.
    
    Args:
        root_dir (str): Root directory containing dataset folders.
        datasets (dict): Dictionary containing dataset configurations.
        metadata_folder (str): Directory to save the metadata file.
        custom_split_code (str): Custom split code for referencing meta filenames.
        num_folds (int): Number of folds for cross-validation.
        seed (int): Random seed used for reproducibility.

    Returns:
        str: Path to the generated metadata JSON file.
    """
    random.seed(seed)
    metadata = {
        "destination_folder": custom_split_code,
        "num_folds": num_folds,
        "datasets": datasets,
        "seed": seed,
        "split_data": {}
    }
    
    total_images = {f"fold{i}": {"train": 0, "val": 0} for i in range(num_folds)}
    
    for dataset_name, dataset_info in datasets.items():
        dataset_path = os.path.join(root_dir, dataset_name, "images")
        image_files = helper.get_image_files(dataset_path)
        
        # Apply percentage filter if less than 100%
        percentage = dataset_info["percentage"]
        if percentage < 100:
            selected_count = int(len(image_files) * (percentage / 100))
            image_files = random.sample(image_files, selected_count)
        
        if dataset_info.get("cross_val", False):

            # Create K-fold cross-validation splits
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

            # Iterate over each fold
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_files)):
                train_files = [image_files[idx] for idx in train_idx]
                val_files = [image_files[idx] for idx in val_idx]

                train_fraction, val_fraction = len(train_files) / len(image_files), len(val_files) / len(image_files)
                metadata["datasets"][dataset_name]["split_ratio"] = {"train": train_fraction, "val": val_fraction}
                metadata["split_data"].setdefault(dataset_name, {})[f"fold{fold_idx}"] = {
                    "train": train_files,
                    "val": val_files
                }
                total_images[f"fold{fold_idx}"]["train"] += len(train_files)
                total_images[f"fold{fold_idx}"]["val"] += len(val_files)
        else:
            split_ratio = dataset_info["split_ratio"]
            temp_split_ratio = {"train": split_ratio["train"], "val": split_ratio["val"], "test": 0.0}
            partitions = helper.split_dataset(image_files, temp_split_ratio)

            metadata["split_data"][dataset_name] = {
                "train": partitions["train"],
                "val": partitions["val"]
            }
            for fold_idx in range(num_folds):
                total_images[f"fold{fold_idx}"]["train"] += len(partitions["train"])
                total_images[f"fold{fold_idx}"]["val"] += len(partitions["val"])

    # Compute average over folds
    avg_train = sum(f["train"] for f in total_images.values()) / num_folds
    avg_val = sum(f["val"] for f in total_images.values()) / num_folds

    total = avg_train + avg_val
    if total > 0:
        metadata["final_split_details"] = total_images
        metadata["final_split_count"] = {"train": round(avg_train, 2), "val": round(avg_val, 2)}
        metadata["final_split_ratio"] = {"train": round(avg_train/total, 2), "val": round(avg_val/total, 2)}
        

    # Output result
    {"train": round(avg_train, 2), "val": round(avg_val, 2)}

    # Save to JSON file
    os.makedirs(metadata_folder, exist_ok=True)
    metadata_file = os.path.join(metadata_folder, f"{custom_split_code}.json")
    helper.write_to_json(metadata, metadata_file)

    return metadata_file



def run_dataset_splitter(datasets_root_dir):
    """Function to execute dataset partitioning and metadata generation."""
    metadata_folder = "experiments/input/custom_splits"
    
    datasets = {
        "LVVO_1k": {
            "percentage": 100,  # Use 100% of the dataset
            "split_ratio": {"train": 0.80, "val": 0.20, "test": 0.0},
            "label_folder": "labels"
        },
        "LVVO_3k": {
            "percentage": 100,
            "split_ratio": {"train": 1.00, "val": 0.0, "test": 0.0},
            "label_folder": "labels"
        },
        # "ldd_dataset": {
        #     "percentage": 100,
        #     "split_ratio": {"train": 0.80, "val": 0.20, "test": 0.0},
        #     "label_folder": "labels"
        # },
        # "lpm_dataset": {
        #     "percentage": 100,
        #     "split_ratio": {"train": 0.80, "val": 0.20, "test": 0.0},
        #     "label_folder": "labels"
        # }
    }
    custom_split_code = "LVVO_4k_val200_seed42"     # Use this name to reference this splitted dataset for training
    metadata_file_path = create_experiment_metadata(datasets_root_dir, datasets, metadata_folder, custom_split_code, seed=42)
    print(f"Metadata JSON file created: {metadata_file_path}")

def run_crossval_dataset_splitter(datasets_root_dir):
    """Function to execute cross-validation dataset partitioning and metadata generation."""
    metadata_folder = "experiments/input/custom_splits"
    
    datasets = {
        "LVVO_1k": {
            "percentage": 100,
            "cross_val": True,
            "split_ratio": None,
            "label_folder": "labels"
        },
        "LVVO_3k": {
            "percentage": 100,
            "cross_val": False,
            "split_ratio": {"train": 100.0, "val": 0.0},
            "label_folder": "labels"
        }
    }
    custom_split_code = "LVVO_4k_val200_cv5_seed42"     # Use this name to reference this splitted dataset for training
    metadata_file_path = create_crossval_experiment_metadata(datasets_root_dir, datasets, metadata_folder, custom_split_code, num_folds=5, seed=42)
    print(f"Cross-validation metadata JSON file created: {metadata_file_path}")




# dataset folders: LVVO_1k_withCategories, LVVO_1k, LVVO_3k, ldd_vdataset, lpm_dataset
if __name__ == "__main__":
    helper = Helper()
    datasets_root_dir = "data/processed"

    # Make sure all datasets have same categories ids and unique image ids in the dataset_info.json files
    run_dataset_splitter(datasets_root_dir)
    run_crossval_dataset_splitter(datasets_root_dir)