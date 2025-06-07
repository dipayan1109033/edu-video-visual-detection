import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torchvision.models import detection

from src.utils.train_utils import *




def get_first_digit(value):
    # Convert the number to a string
    str_value = str(abs(value))  # Use abs() to handle negative numbers
    # Return the first digit or 0 if it's a one-digit number
    return int(str_value[0]) if len(str_value) == 2 else 0

def get_last_digit(value):
    return abs(value) % 10  # Use abs to ensure it works for negative numbers too

def get_faster_rcnn_model(model_code=33, num_classes=2):
    backbone_code = get_first_digit(model_code)
    layer_code = get_last_digit(model_code)

    # Model_code format::: 1st digit [no pretrained = blank/0, pretrained backbone v.1 = 1, backbone v.2 = 2, whole model v.1 = 3, whole model v.2 = 4] and 2nd digit [number of trainable backbone]
    if backbone_code == 1:                          # pretrained backbone v.1 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V1', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 2:                        # pretrained backbone v.2 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V2', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 3:                        # pretrained whole model v.1, finetune starts with [layer_code] trainable backbone layers
        model = detection.fasterrcnn_resnet50_fpn(weights='COCO_V1', trainable_backbone_layers=layer_code)
    elif backbone_code == 4:                        # pretrained whole model v.2, finetune starts with [layer_code] trainable backbone layers
        model = detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1', trainable_backbone_layers=layer_code)
    else:                                           # no pretrained (backbone, whole), training all layers                                      
        model = detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)

    # Modify model architecture for different target classes with using COCO pretrained weights (originally 91 classes)
    if backbone_code in [3, 4] and num_classes != 91:
        # Get the input features for the box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the box predictor head in the model
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def get_mask_rcnn_model(model_code=33, num_classes=2):
    backbone_code = get_first_digit(model_code)
    layer_code = get_last_digit(model_code)

    # Model_code format::: 1st digit [no pretrained = blank/0, pretrained backbone v.1 = 1, backbone v.2 = 2, whole model v.1 = 3, whole model v.2 = 4] and 2nd digit [number of trainable backbone]
    if backbone_code == 1:                          # pretrained backbone v.1 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V1', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 2:                        # pretrained backbone v.2 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V2', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 3:                        # pretrained whole model v.1, finetune starts with [layer_code] trainable backbone layers
        model = detection.maskrcnn_resnet50_fpn(weights='COCO_V1', trainable_backbone_layers=layer_code)
    elif backbone_code == 4:                        # pretrained whole model v.2, finetune starts with [layer_code] trainable backbone layers
        model = detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1', trainable_backbone_layers=layer_code)
    else:                                           # no pretrained (backbone, whole), training all layers                                      
        model = detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)

    # Modify model architecture for different target classes with using COCO pretrained weights (originally 91 classes)
    if backbone_code in [3, 4] and num_classes != 91:
        # Update the box predictor for classification
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        # Update the mask predictor for segmentation
        # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # hidden_layer = 256  # Default value for hidden layers in mask head
        # model.roi_heads.mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        model.roi_heads.mask_predictor = None


    return model

def get_ssd300_model(model_code=33, num_classes=2):
    backbone_code = get_first_digit(model_code)
    layer_code = get_last_digit(model_code)

    # Model_code format::: 1st digit [no pretrained = blank/0, pretrained backbone v.1 = 1, backbone v.2 = 2, whole model v.1 = 3, whole model v.2 = 4] and 2nd digit [number of trainable backbone]
    if backbone_code == 1:                          # pretrained backbone v.1 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.ssd300_vgg16(weights=None, weights_backbone='IMAGENET1K_V1', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 3:                        # pretrained whole model, finetune starts with [layer_code] trainable backbone layers
        model = detection.ssd300_vgg16(weights='COCO_V1', trainable_backbone_layers=layer_code)
    else:                                           # no pretrained (backbone, whole), training all layers                                      
        model = detection.ssd300_vgg16(weights=None, weights_backbone=None, num_classes=num_classes)

    # Modify model architecture for different target classes with using COCO pretrained weights (originally 91 classes)
    if backbone_code in [3] and num_classes != 91: 
        # Get num_anchors from the original model's anchor generator
        num_anchors = []
        for boxes_per_location in model.anchor_generator.num_anchors_per_location():
            num_anchors.append(boxes_per_location)
        
        # Get in_channels from the original model's classification head
        in_channels = []
        for module in model.head.classification_head.module_list:
            in_channels.append(module.in_channels)

        # Create new classification head
        classification_head = detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        # Replace the classification head
        model.head.classification_head = classification_head

    return model

def get_retinanet_model(model_code=33, num_classes=2):
    backbone_code = get_first_digit(model_code)
    layer_code = get_last_digit(model_code)

    # Model_code format::: 1st digit [no pretrained = blank/0, pretrained backbone v.1 = 1, backbone v.2 = 2, whole model v.1 = 3, whole model v.2 = 4] and 2nd digit [number of trainable backbone]
    if backbone_code == 1:                          # pretrained backbone v.1 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.retinanet_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V1', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 2:                        # pretrained backbone v.2 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.retinanet_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V2', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 3:                        # pretrained whole model v.1, finetune starts with [layer_code] trainable backbone layers
        model = detection.retinanet_resnet50_fpn(weights='COCO_V1', trainable_backbone_layers=layer_code)
    elif backbone_code == 4:                        # pretrained whole model v.2, finetune starts with [layer_code] trainable backbone layers
        model = detection.retinanet_resnet50_fpn_v2(weights='COCO_V1', trainable_backbone_layers=layer_code)
    else:                                           # no pretrained (backbone, whole), training all layers                                      
        model = detection.retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)

    # Modify model architecture for different target classes with using COCO pretrained weights (originally 91 classes)
    if backbone_code in [3, 4] and num_classes != 91:
        # Get the number of anchors per location
        num_anchors = model.head.classification_head.num_anchors
        
        # Get the input channels of the classification head
        in_channels = model.head.classification_head.cls_logits.in_channels

        # Instantiate the new classification head with the new number of classes
        new_classification_head = detection.retinanet.RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            prior_probability=0.01
        )

        # Replace the current classification head with the new one
        model.head.classification_head = new_classification_head
        model.head.classification_head.num_classes = num_classes

    return model

def get_fcos_model(model_code=33, num_classes=2):
    backbone_code = get_first_digit(model_code)
    layer_code = get_last_digit(model_code)

    # Model_code format::: 1st digit [no pretrained = blank/0, pretrained backbone v.1 = 1, backbone v.2 = 2, whole model v.1 = 3, whole model v.2 = 4] and 2nd digit [number of trainable backbone]
    if backbone_code == 1:                          # pretrained backbone v.1 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.fcos_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V1', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 2:                        # pretrained backbone v.2 only, finetune/training starts with [layer_code] trainable backbone layers
        model = detection.fcos_resnet50_fpn(weights=None, weights_backbone='IMAGENET1K_V2', trainable_backbone_layers=layer_code, num_classes=num_classes)
    elif backbone_code == 3:                        # pretrained whole model, finetune starts with [layer_code] trainable backbone layers
        model = detection.fcos_resnet50_fpn(weights='COCO_V1', trainable_backbone_layers=layer_code)
    else:                                           # no pretrained (backbone, whole), training all layers                                      
        model = detection.fcos_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)

    # Modify model architecture for different target classes with using COCO pretrained weights (originally 91 classes)
    if backbone_code in [3] and num_classes != 91:
        # Get the number of anchors per location
        num_anchors = model.head.classification_head.num_anchors
        
        # Get the input channels of the classification head
        in_channels = model.head.classification_head.cls_logits.in_channels
        
        # Instantiate the new classification head with the new number of classes
        new_classification_head = detection.fcos.FCOSClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            prior_probability=0.01
        )

        # Replace the current classification head with the new one
        model.head.classification_head = new_classification_head
        model.head.classification_head.num_classes = num_classes

    return model

def get_pytorch_model(model_identifier, model_code=33, num_classes=2):
    model_identifier = model_identifier.lower()
    if model_identifier == "rcnn":
        model = get_faster_rcnn_model(model_code = model_code, num_classes = num_classes)
    elif model_identifier == "maskrcnn":
        model = get_mask_rcnn_model(model_code = model_code, num_classes = num_classes)
    elif model_identifier == "fcos":
        model = get_fcos_model(model_code = model_code, num_classes = num_classes)
    elif model_identifier == "retinanet":
        model = get_retinanet_model(model_code = model_code, num_classes = num_classes)
    elif model_identifier == "ssd":
        model = get_ssd300_model(model_code = model_code, num_classes = num_classes)
    else:
        print("Please choose a model name from 'rcnn', 'maskrcnn', 'fcos', 'retinanet', 'ssd' ")
    return model



# For "StepLR", provide 'lr_step_size' and 'lr_gamma'
# For "ExponentialLR", provide 'lr_gamma'
# For "ReduceLROnPlateau", provide 'lr_factor' and 'lr_patience'
# For "OneCycleLR", no additional requirement 

def get_lr_scheduler(optimizer, cfg, steps_per_epoch):

    if cfg.train.scheduler.type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler.step_size, gamma=cfg.train.scheduler.gamma)
    elif cfg.train.scheduler.type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.train.scheduler.gamma)
    elif cfg.train.scheduler.type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6, factor=cfg.train.scheduler.factor, patience=cfg.train.scheduler.patience)
    elif cfg.train.scheduler.type == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=steps_per_epoch, epochs=cfg.train.epoch, max_lr=cfg.train.lr, pct_start=0.0, div_factor=1.0, final_div_factor=1e4)
    else:
        raise ValueError(f"Invalid learning rate scheduler: {cfg.train.scheduler.type}. Choose from StepLR, ExponentialLR, ReduceLROnPlateau, OneCycleLR.")

    return scheduler


