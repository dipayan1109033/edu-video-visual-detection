
import os
import math
import shutil
from ultralytics import YOLO

from src.utils.train_utils import *
from src.utils.common import Helper
helper = Helper()




def save_predictions_to_custom_labels(results, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    def truncate_float(value, decimals=4):
        factor = 10 ** decimals
        return math.trunc(float(value) * factor) / factor

    # Process the predictions
    for idx, result in enumerate(results):
        image_filename = os.path.basename(result.path)
        height, width = result.orig_shape

        # Initialize the custom labels structure
        formatted_output = {
            "asset": {
                "name": image_filename,
                "image_id": idx+1,
                "size": {
                    "width": width,
                    "height": height
                }
            },
            "objects": []
        }
        boxes = result.boxes.xyxy   # Bounding boxes (xmin, ymin, xmax, ymax)
        labels = result.boxes.cls   # Class labels
        scores = result.boxes.conf  # Get confidence scores

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(lambda x: truncate_float(x, decimals=4), box)
            obj_class = int(labels[i])                # Class label
            conf_score = round(float(scores[i]), 4)   # Convert to float

            # Calculate width and height of the bounding box
            bbox_width = truncate_float(xmax - xmin, decimals=4) 
            bbox_height = truncate_float(ymax - ymin, decimals=4)

            # Add the bounding box information to the output
            formatted_output["objects"].append({
                "class": obj_class,
                "score": conf_score,
                "boundingBox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "width": bbox_width,
                    "height": bbox_height
                }
            })

        json_filepath = os.path.join(output_folder_path, f"{os.path.splitext(image_filename)[0]}.json")
        helper.write_to_json(formatted_output, json_filepath)

def predict_using_model(cfg, batch_size = 10):
    train_root_dir = f"{cfg.path.output_dir}/train"
    # Load a pretrained YOLO model
    if cfg.model.saved_model_folder:
        saved_model_path = f"{train_root_dir}/{cfg.model.saved_model_folder}/weights/best.pt"
        model = YOLO(saved_model_path)
        print(f"Loaded pretrained model from: {saved_model_path}")
    else:
        #cfg.model.pretrained_model = "experiments/output/yolo/train/crossval29/fold0/weights/best.pt"
        model = YOLO(cfg.model.pretrained_model)
        print(f"Loaded pretrained model from: {cfg.model.pretrained_model}")
        #raise ValueError(f"You need to provide non-zero yolo trained number to load the trained model")


    # Image folder for predictions
    image_folderpath = cfg.path.predict_image_folder
    if not image_folderpath:
        raise ValueError(f"Image folder path need to be specified through: cfg.path.predict_image_folder")

    # Output folder for storing predictions
    predict_root_dir = f"{cfg.path.output_dir}/predict/{cfg.model.saved_model_folder}"
    image_folder = helper.get_immediate_folder_name(image_folderpath)
    output_folder_path = os.path.join(predict_root_dir, image_folder)

    # Predictions
    results = model.predict(image_folderpath, device='mps', batch=batch_size)
    save_predictions_to_custom_labels(results, output_folder_path)


def validate_model_one_fold(cfg, model, dataset_yaml, val_root_path, dataset_dir, dataset_folder, fold_idx):
    val_root_folder = helper.get_immediate_folder_name(val_root_path)

    # Validate the model
    results = model.val(data=dataset_yaml, split='val', save_json= True, project=val_root_path, name=f"fold{fold_idx}")

    # Save predicted bounding boxes and some metrics
    val_output_path = os.path.join(val_root_path, f"fold{fold_idx}")
    save_filename = f"{val_root_folder}_{cfg.exp.save_name}_fold{fold_idx}__{dataset_folder}.json"
    output_filepath = save_predictions_forYOLO(cfg, results, val_output_path, dataset_dir, dataset_folder, 'val', save_filename, extension=".jpg", folds=f"fold{fold_idx}")
    
    dataset_fold_dir = os.path.join(dataset_dir, f"fold{fold_idx}")
    evaluate_predictions(dataset_fold_dir, 'val', output_filepath, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier, csvFlag=False)
    return output_filepath

def crossvalidate_model(cfg):
    dataset_dir = cfg.path.temp_dataset_path
    dataset_folder = helper.get_immediate_folder_name(dataset_dir)

    # Clean up the output folder for storing validation output
    val_root_path = f"{cfg.path.output_dir}/validate/crossval{cfg.exp.number}"
    if os.path.exists(val_root_path): shutil.rmtree(val_root_path)
    os.makedirs(val_root_path, exist_ok=True)

    # Iterate over each fold
    predictions_filelist = []
    for fold_idx in range(cfg.data.num_folds):

        # Load a pretrained YOLO model
        if cfg.model.saved_model_folder:
            saved_model_path = f"{cfg.path.output_dir}/train/{cfg.model.saved_model_folder}/fold{fold_idx}/weights/best.pt"
            model = YOLO(saved_model_path)
            print(f"Loaded pretrained model from: {saved_model_path}")
        else:
            model = YOLO(cfg.model.pretrained_model)

        # Yolo dataset yaml filepath
        dataset_yaml = os.path.join(dataset_dir, f"fold{fold_idx}_yolo_dataset.yaml")

        # Train the model
        train_root_dir = f"{cfg.path.output_dir}/train/crossval{cfg.exp.number}"
        model.train(data=dataset_yaml, 
                    seed=cfg.exp.seed, 
                    batch=cfg.train.batch_size, 
                    epochs=cfg.train.epoch, 
                    lr0=cfg.train.lr, 
                    optimizer=cfg.train.optimizer, 
                    project=train_root_dir, 
                    name=f"fold{fold_idx}",
                    deterministic=cfg.train.deterministic,
                    freeze=cfg.train.freeze_layers,
                    amp=False,
                    mosaic=0.0,
                    mixup=0.0,
                    auto_augment=None,
                    erasing=0.0
                )
        # Save config arguments
        train_root_path = os.path.join(train_root_dir, f"fold{fold_idx}")
        save_OmegaConfig(cfg, train_root_path)
        
        # Load trained model
        model_savepath = f"{train_root_path}/weights/best.pt"
        model = YOLO(model_savepath)  

        # Test the trained model
        output_filepath = validate_model_one_fold(cfg, model, dataset_yaml, val_root_path, dataset_dir, dataset_folder, fold_idx)
        predictions_filelist.append(output_filepath)

    evaluate_cv_predictions(predictions_filelist, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier)


def validate_model(cfg, model, train_root_path, dataset_dir, dataset_folder, split="val", exp_save_name=None):
    train_folder = helper.get_immediate_folder_name(train_root_path)

    if exp_save_name is None: exp_save_name = get_exp_save_name(train_root_path)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Temp dataset setup missing for : {dataset_dir}")

    # Clean up the output folder for storing validation output
    val_root_dir = f"{cfg.path.output_dir}/validate/test{cfg.exp.number}"
    if os.path.exists(val_root_dir): shutil.rmtree(val_root_dir)
    
    # Validate the model
    dataset_yaml = os.path.join(dataset_dir, "yolo_dataset.yaml")
    results = model.val(data=dataset_yaml, 
                        split=split, 
                        save_json= True, 
                        project=val_root_dir, 
                        name=dataset_folder
                    )

    # Save predicted bounding boxes and some metrics
    val_output_path = os.path.join(val_root_dir, dataset_folder)
    save_filename = f"{train_folder}_{exp_save_name}__{dataset_folder}.json"
    output_filepath = save_predictions_forYOLO(cfg, results, val_output_path, dataset_dir, dataset_folder, split, save_filename, extension=".jpg")

    evaluate_predictions(dataset_dir, split, output_filepath, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier)

def train_model(cfg):
    dataset_dir = cfg.path.temp_dataset_path
    dataset_folder = helper.get_immediate_folder_name(dataset_dir)

    # Load a pretrained YOLO model
    train_root_dir = f"{cfg.path.output_dir}/train"
    if cfg.model.saved_model_folder:
        saved_model_path = f"{train_root_dir}/{cfg.model.saved_model_folder}/weights/best.pt"
        model = YOLO(saved_model_path)
        print(f"Loaded pretrained model from: {saved_model_path}")
    else:
        model = YOLO(cfg.model.pretrained_model)

    # Yolo dataset yaml filepath
    dataset_yaml = os.path.join(dataset_dir, "yolo_dataset.yaml")

    # Train the model
    model.train(data=dataset_yaml, 
                seed=cfg.exp.seed, 
                batch=cfg.train.batch_size, 
                epochs=cfg.train.epoch, 
                lr0=cfg.train.lr, 
                optimizer=cfg.train.optimizer, 
                project=train_root_dir, 
                name=f"train{cfg.exp.number}", 
                save_period=cfg.train.save_interval,
                deterministic=cfg.train.deterministic,
                freeze=cfg.train.freeze_layers,
                amp=False,
                mosaic=0.0,
                mixup=0.0,
                auto_augment=None,
                erasing=0.0
            )
    # Save config arguments
    train_root_path = f"{train_root_dir}/train{cfg.exp.number}"
    save_OmegaConfig(cfg, train_root_path)

    # Load trained model
    model_savepath = f"{train_root_path}/weights/best.pt"
    model = YOLO(model_savepath)  


    # Test the trained model
    validate_model(cfg, model, train_root_path, dataset_dir, dataset_folder, split=cfg.data.test_split, exp_save_name=cfg.exp.save_name)



def test_model(cfg):
    dataset_dir = cfg.path.temp_dataset_path
    dataset_folder = helper.get_immediate_folder_name(dataset_dir)

    train_root_dir =  os.path.join(cfg.path.output_dir, "train")

    # Load a pretrained YOLO model
    if cfg.model.saved_model_folder:
        if "train" in cfg.model.saved_model_folder or "fold" in cfg.model.saved_model_folder:
            saved_model_path = f"{train_root_dir}/{cfg.model.saved_model_folder}/weights/best.pt"
            model = YOLO(saved_model_path)
            print(f"Loaded saved pretrained model from: {saved_model_path}")

            # Test the trained model
            train_root_path = os.path.join(train_root_dir, cfg.model.saved_model_folder)
            validate_model(cfg, model, train_root_path, dataset_dir, dataset_folder, split=cfg.data.test_split, exp_save_name=cfg.exp.save_name)

        elif "crossval" in cfg.model.saved_model_folder:
            # Clean up the output folder for storing validation output
            val_root_path = os.path.join(cfg.path.output_dir, "validate", cfg.model.saved_model_folder)
            if os.path.exists(val_root_path): shutil.rmtree(val_root_path)
            os.makedirs(val_root_path, exist_ok=True)

            predictions_filelist = []
            for fold_idx in range(cfg.data.num_folds):
                saved_model_path = f"{train_root_dir}/{cfg.model.saved_model_folder}/fold{fold_idx}/weights/best.pt"
                model = YOLO(saved_model_path)
                print(f"Loaded pretrained model from: {saved_model_path}")

                # Validate the model for one fold
                dataset_yaml = os.path.join(dataset_dir, f"fold{fold_idx}_yolo_dataset.yaml")
                output_filepath = validate_model_one_fold(cfg, model, dataset_yaml, val_root_path, dataset_dir, dataset_folder, fold_idx)
                predictions_filelist.append(output_filepath)

            evaluate_cv_predictions(predictions_filelist, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier)
    else:
        raise ValueError(f"You need to provide saved_model_folder name to load the trained model")





def main(cfg):
    # Set random seeds for this process
    set_seeds(cfg.exp.seed)
    cfg.exp.device = check_for_cuda_device(cfg.exp.device)

    # Get output folder number for the experiemnt
    if cfg.exp.mode == 'train':
        train_root_dir = f"{cfg.path.output_dir}/train"
        cfg.exp.number = get_new_folder_num(train_root_dir, prefix="train")
    elif cfg.exp.mode == 'crossval':
        train_root_dir = f"{cfg.path.output_dir}/train"
        cfg.exp.number = create_unique_experiment_folder(train_root_dir, prefix="crossval")


    if cfg.exp.mode == 'train':
        train_model(cfg)
    elif cfg.exp.mode == 'crossval':
        crossvalidate_model(cfg)
    elif cfg.exp.mode == 'test':
        test_model(cfg)
    elif cfg.exp.mode == "predict":
        predict_using_model(cfg, batch_size = 10)
    else:
        raise ValueError(f"Unsupported experiment mode: {cfg.exp.mode}. Supported modes are 'train', 'crossval', 'test', and 'predict'.")