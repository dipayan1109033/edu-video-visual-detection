import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from src.dataset.dataset import LectureFrameODDataset
from src.models.torchvision_models import get_pytorch_model, get_lr_scheduler
from src.utils.train_utils import *
from src.utils.common import Helper
helper = Helper()




def train_torchvisionModel(cfg, model, optimizer, train_loader, val_loader = None, save_root_dir=None, fold_tag=""):
    # keys = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]   # FasterRCNN loss names.
    loss_dict = {'train_loss': [], 'valid_loss': [], 'smooth_train_loss': [], 'smooth_valid_loss': []}

    device = torch.device(cfg.exp.device)
    lr_scheduler = get_lr_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    for epoch in range(cfg.train.epoch):
        print("----------Epoch {}----------".format(epoch+1))

        # Train the model for one epoch
        train_loss_list = train_one_epoch(model, optimizer, train_loader, device, epoch)
        loss_dict['train_loss'].extend(train_loss_list)
        loss_dict['smooth_train_loss'].append(sum(train_loss_list) / len(train_loss_list))

        if val_loader is not None:
            # Run evaluation
            valid_loss_list = evaluate(model, val_loader, device)
            loss_dict['valid_loss'].extend(valid_loss_list)
            loss_dict['smooth_valid_loss'].append(sum(valid_loss_list) / len(valid_loss_list))

        if cfg.train.scheduler.type == "ReduceLROnPlateau":
            lr_scheduler.step(loss_dict['smooth_valid_loss'][-1])
        else:
            lr_scheduler.step()

        if (epoch+1) % cfg.train.save_interval == 0 and len(fold_tag)==0:
            save_model_checkpoint(model, optimizer, epoch+1, loss_dict, save_root_dir, save_filename=f"epoch{epoch+1}.pt")

    # Save the trained model
    save_model_checkpoint(model, optimizer, cfg.train.epoch, loss_dict, save_root_dir, save_filename="last.pt")

    save_path = os.path.join(save_root_dir, "logs.json")
    helper.write_to_json(loss_dict, save_path)

    plot_learning_curves(loss_dict['smooth_train_loss'], loss_dict['smooth_valid_loss'], cfg.model.identifier, save_root_dir)
    print("Training complete!")

    return model

def test_torchvisionModel(cfg, model, dataset_dir, dataset_folder, split, save_filename, fold_tag=""):
    device = torch.device(cfg.exp.device)

    # Validation dataset.
    test_dataset_path = os.path.join(dataset_dir, fold_tag, split)
    test_dataset = LectureFrameODDataset(test_dataset_path, transformFlag = False)

    # Create the DataLoaders from the Datasets. 
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size = cfg.train.batch_size, shuffle = False, collate_fn = collate_fn)

    # Make predictions and get ground truth targets
    predictions, targets = predict(model, test_dataset, device)

    predictions = filter_predictions(predictions, confidence_threshold=0.0, iou_threshold=cfg.model.nms_threshold)
    predictions_filepath = save_predictions(cfg, predictions, targets, dataset_dir, dataset_folder, save_filename)
    print("Test complete!")

    return predictions_filepath



def crossvalidate_model(cfg):
    print("In plain_detection_models :> crossvalidate_model():")
    dataset_dir = cfg.path.temp_dataset_path
    dataset_folder = cfg.data.folder

    # Iterate over each fold
    predictions_filelist = []
    for fold_idx in range(cfg.data.num_folds):
        dataset_fold_dir = os.path.join(dataset_dir, f"fold{fold_idx}")

        # Train dataset: Set transformFlag = True to apply transforms to the training images
        train_dataset_path = os.path.join(dataset_dir, f"fold{fold_idx}", "train")
        train_dataset = LectureFrameODDataset(train_dataset_path, transformFlag = cfg.data.use_augmentation)

        # Validation dataset
        val_dataset_path = os.path.join(dataset_dir, f"fold{fold_idx}", "val")
        val_dataset = LectureFrameODDataset(val_dataset_path, transformFlag = False)

        # Create the DataLoaders from the Datasets
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.train.batch_size, shuffle = True, collate_fn = collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = cfg.train.batch_size, shuffle = False, collate_fn = collate_fn)
        

        # Create the pytorch model
        model = get_pytorch_model(cfg.model.identifier, cfg.model.code, cfg.data.num_classes)

        # Use the stochastic gradient descent optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr = cfg.train.lr, momentum = cfg.train.momentum,  weight_decay = cfg.train.weight_decay)

        # # Load saved model if provided
        # if args.saved_model_name:
        #     saved_model_path = os.path.join(cfg.path.output_dir, "saved_models", args.saved_model_name)
        #     model, optimizer,  _,  _ = load_model_checkpoint(saved_model_path, model, optimizer)

        # Train the model over number of epochs
        save_root_dir = os.path.join(cfg.path.output_dir, "crossval", f"crossval{cfg.exp.number}", f"fold{fold_idx}")
        model = train_torchvisionModel(cfg, model, optimizer, train_loader, val_loader, save_root_dir, fold_tag=f"fold{fold_idx}_")
        save_OmegaConfig(cfg, save_root_dir)

        # Validate the trained model
        save_filename = f"crossval{cfg.exp.number}_{cfg.exp.save_name}_fold{fold_idx}__{dataset_folder}.json"
        predictions_filepath = test_torchvisionModel(cfg, model, dataset_dir, dataset_folder, split="val", save_filename=save_filename, fold_tag=f"fold{fold_idx}")
        dataset_fold_dir = os.path.join(dataset_dir, f"fold{fold_idx}")
        evaluate_predictions(dataset_fold_dir, 'val', predictions_filepath, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier, csvFlag=False)
        predictions_filelist.append(predictions_filepath)

    evaluate_cv_predictions(predictions_filelist, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier)


def train_model(cfg):
    print("In plain_detection_models :> train_model():")
    dataset_dir = cfg.path.temp_dataset_path
    dataset_folder = cfg.data.folder

    # Train dataset: Set transformFlag = True to apply transforms to the training images
    train_dataset_path = os.path.join(dataset_dir, "train")
    train_dataset = LectureFrameODDataset(train_dataset_path, transformFlag = cfg.data.use_augmentation)

    # Validation dataset
    val_dataset_path = os.path.join(dataset_dir, "val")
    val_dataset = LectureFrameODDataset(val_dataset_path, transformFlag = False)

    # Create the DataLoaders from the Datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.train.batch_size, shuffle = True, collate_fn = collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = cfg.train.batch_size, shuffle = False, collate_fn = collate_fn)
    

    # Create the pytorch model
    model = get_pytorch_model(cfg.model.identifier, cfg.model.code, cfg.data.num_classes)

    # Use the stochastic gradient descent optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = cfg.train.lr, momentum = cfg.train.momentum,  weight_decay = cfg.train.weight_decay)

    # # Load saved model if provided
    # if args.saved_model_name:
    #     saved_model_path = os.path.join(cfg.path.output_dir, "saved_models", args.saved_model_name)
    #     model, optimizer,  _,  _ = load_model_checkpoint(saved_model_path, model, optimizer, args.device)

    # Train the model over number of epochs
    save_root_dir = os.path.join(cfg.path.output_dir, "train", f"train{cfg.exp.number}")
    model = train_torchvisionModel(cfg, model, optimizer, train_loader, val_loader, save_root_dir)
    save_OmegaConfig(cfg, save_root_dir)

    # Test the trained model
    save_filename = f"train{cfg.exp.number}_{cfg.exp.save_name}__{dataset_folder}.json"
    predictions_filepath = test_torchvisionModel(cfg, model, dataset_dir, dataset_folder, split = "val", save_filename=save_filename)
    evaluate_predictions(dataset_dir, cfg.data.test_split, predictions_filepath, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier)


def test_model(cfg):
    dataset_dir = cfg.path.temp_dataset_path
    dataset_folder = cfg.data.folder

    train_root_dir = f"{cfg.path.output_dir}/train"
    if cfg.exp.number == 0:
        raise ValueError(f"You need to provide non-zero experiment number to load the trained model")

    # Retrieve training parameters
    train_root_path = f"{train_root_dir}/train{cfg.exp.number}"
    my_config = read_OmegaConfig(train_root_path)

    # Load trained model
    model = get_pytorch_model(my_config.model.identifier, my_config.model.code, my_config.data.num_classes)
    ckpt_file_path = f"{train_root_path}/checkpoints/last.pt"
    checkpoint = torch.load(ckpt_file_path, map_location=cfg.exp.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test the trained model
    save_filename = f"train{cfg.exp.number}_{my_config.exp.save_name}__{dataset_folder}.json"
    predictions_filepath = test_torchvisionModel(cfg, model, dataset_dir, dataset_folder, split = "val", save_filename=save_filename)
    evaluate_predictions(dataset_dir, cfg.data.test_split, predictions_filepath, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier)




def main(cfg):
    # Set random seeds for this process
    set_seeds(cfg.exp.seed)
    cfg.exp.device = check_for_cuda_device(cfg.exp.device)

    # Get output folder number for the experiemnt
    if cfg.exp.mode == 'train':
        train_root_dir = f"{cfg.path.output_dir}/train"
        cfg.exp.number = create_unique_experiment_folder(train_root_dir, prefix="train")
    elif cfg.exp.mode == 'crossval':
        crossval_root_dir = f"{cfg.path.output_dir}/crossval"
        cfg.exp.number = create_unique_experiment_folder(crossval_root_dir, prefix="crossval")


    if cfg.exp.mode == 'train':
        train_model(cfg)
    elif cfg.exp.mode == 'crossval':
        crossvalidate_model(cfg)
    elif cfg.exp.mode == 'test':
        test_model(cfg)
    else:
        raise ValueError(f"Unsupported experiment mode: {cfg.exp.mode}. Supported modes are 'train', 'crossval', and 'test'.")