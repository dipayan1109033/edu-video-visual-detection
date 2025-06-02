import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings, logging
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=ResourceWarning)

import torch
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

import src.yolo_model as yolo_model
import tvdetection_models as tvdetection_models

from src.utils.data_utils import *
from src.utils.common import Helper
helper = Helper()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocessing(config, debug=False):
    # Set random seeds for reproducibility
    set_seeds(config.exp.seed)

    # Update dataset and project directory based on local/server
    if os.path.exists(config.path.project_root_dir.server):
        config.exp.on_server = True
        config.data.root_dir = config.data.root_dir.server
        config.path.project_root_dir = config.path.project_root_dir.server
    else:
        config.exp.on_server = False
        config.data.root_dir = config.data.root_dir.local
        config.path.project_root_dir = config.path.project_root_dir.local

    # Add few arguments
    if config.model.identifier == "yolo":
        config.exp.save_name = f"rs{config.exp.seed}_{config.model.identifier}_{config.exp.name}_b{config.train.batch_size}_e{config.train.epoch}_lr{config.train.lr}"
    else:
        config.exp.save_name = f"rs{config.exp.seed}_{config.model.identifier}{config.model.code}_{config.exp.name}_b{config.train.batch_size}_e{config.train.epoch}_lr{config.train.lr}"

    if debug: print(OmegaConf.to_yaml(config))

    return config

def setup_dataset(cfg):
    """
    Set up the dataset for training, validation, and testing based on the configuration.

    Args:
        cfg (OmegaConf): The configuration object containing dataset and path details.

    Returns:
        str: Path to the prepared dataset directory.
    """
    # Extract dataset folder name and root directory from configuration
    dataset_folder = cfg.data.folder
    src_datasets_root_dir = cfg.data.root_dir

    # Define the source and temporary directory for datasets
    src_dataset_path = os.path.join(src_datasets_root_dir, dataset_folder)
    temp_datasets_root_dir = os.path.join(cfg.path.project_root_dir, cfg.path.input_dir, "temp_datasets")

    # Choose data partitions

    if cfg.exp.mode == "train" or (cfg.model.saved_model_folder and ("train" in cfg.model.saved_model_folder or "fold" in cfg.model.saved_model_folder)):
        if cfg.data.split_code:             # Partition with provided custom splits file
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{cfg.data.split_code}")
            create_experimental_dataset_from_metadata(src_datasets_root_dir, temp_dataset_path, cfg.data.custom_splits_dir, cfg.data.split_code)
        else:                               # Partition with splits percentage ratios
            train_split, val_split, test_split = [int(split * 100) for split in cfg.data.split_ratios]
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{dataset_folder}_{train_split}_{val_split}_{test_split}_seed{cfg.exp.seed}")
            partition_dataset_by_ratio(src_dataset_path, temp_dataset_path, cfg.data.split_ratios, seed=cfg.exp.seed)

    elif cfg.exp.mode == "crossval" or "crossval" in cfg.model.saved_model_folder:      # For cross-val experiement
        if cfg.data.split_code:              # Partition with provided custom splits file
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{cfg.data.split_code}")
            create_experimental_cv_dataset_from_metadata(src_datasets_root_dir, temp_dataset_path, cfg.data.custom_splits_dir, cfg.data.split_code)
        elif cfg.data.use_replacement:       # with replacement
            train_split, val_split, test_split = [int(split * 100) for split in cfg.data.split_ratios]
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{dataset_folder}_{train_split}_{val_split}_{test_split}_cv{cfg.data.num_folds}")
            cv_partition_with_replacement(src_dataset_path, temp_dataset_path, cfg.data.split_ratios, cfg.data.num_folds, seed=cfg.exp.seed)
        else:                               # without replacement
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{dataset_folder}_cv{cfg.data.num_folds}_seed{cfg.exp.seed}")
            cv_partition_without_replacement(src_dataset_path, temp_dataset_path, cfg.data.num_folds, seed=cfg.exp.seed)
            
    # Add the path to the config
    cfg.path.temp_dataset_path = temp_dataset_path

    return cfg


@hydra.main(version_base=None, config_path='../configs', config_name='default_config')
def main(config: DictConfig):
    config = preprocessing(config, debug=True)
    cfg = setup_dataset(config)


    if cfg.model.identifier == "yolo":
        yolo_model.main(cfg)
    elif cfg.model.identifier in ['yolo', 'rcnn', 'maskRCNN', 'fcos', 'retinanet', 'ssd']:
        tvdetection_models.main(cfg)
    else:
        raise ValueError(f"The model identifier '{cfg.model.identifier}' is invalid.")




if __name__ == "__main__":
    main()
