# Visual Content Detection in Educational Videos

📘 *Conference Paper (To Appear)* • [📝 arXiv Paper]() • [📦 Download Datasets](https://github.com/dipayan1109033/LVVO_dataset) • [📄 LVVO Dataset arXiv Paper]()

This repository contains the source code for the research paper:  
**"Visual Content Detection in Educational Videos with Transfer Learning and Dataset Enrichment"**  
by Dipayan Biswas, Shishir Shah, and Jaspal Subhlok (University of Houston)

## 📄 Overview

This work presents a deep learning framework for detecting visual objects—such as tables, charts, images, and illustrations—in lecture video frames using transfer learning and dataset enrichment. Evaluated on three datasets (LDD, LPM, and the newly introduced LVVO), six object detection models were fine-tuned, with YOLOv11 achieving the best performance. The model was further optimized through cross-dataset training and a semi-supervised auto-labeling pipeline, demonstrating that transfer learning and data enrichment significantly improve detection accuracy under limited annotation.


### 🔍 Key Contributions

- Introduced the **LVVO dataset** comprising 4,000 annotated lecture video frames.
- Benchmarked six state-of-the-art object detection models across LVVO, LDD, and LPM datasets.
- Addressed the challenge of generalization through cross-dataset training and analysis on diverse educational video sources.
- Boosted model accuracy with limited labeled data using a **semi-supervised auto-labeling pipeline**.


## 📊 Datasets

We utilize three annotated datasets: **LVVO** (4,000 frames, introduced in this work), **LDD**, and **LPM**.  
🔗 See the [LVVO Dataset Repository](https://github.com/dipayan1109033/LVVO_dataset) for details and downloads.


## ⚙️ Setup

### 1. Clone the repository:
```bash
git clone https://github.com/dipayan1109033/edu-video-visual-detection.git
cd edu-video-visual-detection
```

### 2. Clone the `calculate_ODmetrics` repo inside `src/utils/`:
```bash
cd src/utils
git clone https://github.com/dipayan1109033/calculate_ODmetrics.git
```

### 3. Set up the environment:
Set up a virtual environment and install dependencies from `requirements.txt`.

### 4. Prepare the dataset:
Download the dataset from the [LVVO Dataset Repository](https://github.com/dipayan1109033/LVVO_dataset), place the zip files in `data/processed/`, and unzip them:
```bash
cd data/processed
unzip dataset_name.zip
```

## 🏋️ Model Training

This project supports training both YOLOv11 and torchvision-based models (e.g., Faster R-CNN, RetinaNet, FCOS) using either manual split ratios or predefined dataset splits.  
Predefined splits can be generated using the following script:

```bash
python src/prepare/setup_experiment.py
```

### Key training arguments
<details>
<summary><strong>Click to expand key training arguments</strong></summary>

- **`model.identifier`**: Model name (`yolo`, `rcnn`, `maskrcnn`, `retinanet`, `fcos`, `ssd`)
- **`model.pretrained_model`**: Path or name of pretrained weights (for YOLOv11)
- **`model.code`**: Two-digit code for torchvision models, specifying the backbone and number of frozen layers. See `src/models/torchvision_models.py` for details.
- **`exp.mode`**: Training mode (`"train"` or `"crossval"`)
- **`exp.name`**: User given experiment name (used to save logs and checkpoints)
- **`data.folder`**: Dataset directory name (used with `split_ratios`)
- **`data.split_ratios`**: Train/val/test ratio, e.g., `[0.8,0.2,0.0]`
- **`data.split_code`**: Identifier for a custom dataset split created using `src/prepare/setup_experiment.py` and saved in `experiments/input/custom_splits/`
- **`data.num_folds`**: Number of folds for cross-validation (e.g., `5`)
- **`train.lr`**: Learning rate (e.g., `0.001`)
- **`train.epoch`**: Number of training epochs

➡️ For additional arguments and full configuration options, refer to `configs/experiment.yaml`.

</details>


### Example Commands

✅ YOLOv11 Training with split ratios

```bash
python src/main.py model.identifier="yolo" model.pretrained_model="yolo11m.pt" exp.mode="train" exp.name="train_yolo_LVVO1k" data.folder="LVVO_1k" data.split_ratios="[0.8,0.2,0.0]" train.lr=0.001 train.epoch=30
```
✅ YOLOv11 Training with split code

```bash
python src/main.py model.identifier="yolo" model.pretrained_model="yolo11m.pt" exp.mode="train" exp.name="train_yolo_csplitLVVO4k" data.split_code="LVVO_4k_val200_seed42" train.lr=0.001 train.epoch=30
```

✅ Torchvision Model Cross-validation (e.g., Faster R-CNN)

```bash
python src/main.py model.identifier="rcnn" model.code=33 exp.mode="crossval" exp.name="crossval_rcnn_LVVO1k" data.folder="LVVO_1k" data.num_folds=5 train.lr=0.001 train.epoch=30
```
✅ YOLOv11 Cross-validation with split code

```bash
python src/main.py model.identifier="yolo" model.pretrained_model="yolo11m.pt" exp.mode="crossval" exp.name="crossval_yolo_csplitLVVO4k" data.split_code="LVVO_4k_val200_cv5_seed42" train.lr=0.001 train.epoch=30

```


## 📈 Results

#### 📊 Table 1: AP50% Comparison of Object Detection Models Across Datasets (80%:20% Train-Validation Split)

| Model        | LVVO_1k | LDD    | LPM    |
|--------------|---------|--------|--------|
| SSD          | 83.81   | 87.79  | 85.73  |
| RetinaNet    | 78.34   | 88.82  | 86.92  |
| FCOS         | 83.46   | 89.12  | 87.58  |
| Faster-RCNN  | 85.38   | 88.72  | 87.40  |
| Mask-RCNN    | 85.74   | 89.31  | 86.74  |
| YOLOv11      | 89.45   | 94.29  | 92.08  |

> **Note:** Table 1 reports the numerical results visualized in **Figure 2** of the paper.


#### 📊 Table 2: Comparison of Logiform and YOLOv11 on Classic Metrics (IoU = 0.5, Mean ± Std)


| Model     | Precision (%)     | Recall (%)        | F1 Score (%)      |
|-----------|-------------------|-------------------|-------------------|
| Logiform | 64.33 ± 2.73      | 62.88 ± 3.29      | 63.57 ± 2.67      |
| YOLOv11   | 86.76 ± 1.87      | 83.60 ± 1.56      | 85.14 ± 1.25      |

> **Note:** Table 2 reports the numerical results visualized in **Figure 3** of the paper.


#### 📊 Table 3: Cross-Dataset Performances of YOLOv11 (Mean ± Std, 5-Fold Cross-Validation)

| Training Dataset | Test on LVVO_1k | Test on LDD     | Test on LPM     |
|------------------|------------------|------------------|------------------|
| **AP50 (%)**     |                  |                  |                  |
| LVVO_1k          | 90.95 ± 1.12     | 69.69 ± 2.53     | 74.34 ± 2.50     |
| LDD              | 75.92 ± 1.67     | 93.56 ± 0.77     | 68.83 ± 3.68     |
| LPM              | 80.05 ± 1.62     | 58.66 ± 3.11     | 92.65 ± 0.66     |

| Training Dataset | Test on LVVO_1k | Test on LDD     | Test on LPM     |
|------------------|------------------|------------------|------------------|
| **AP (%)**       |                  |                  |                  |
| LVVO_1k          | 77.93 ± 1.38     | 50.09 ± 2.60     | 50.10 ± 2.31     |
| LDD              | 59.57 ± 1.72     | 87.74 ± 0.58     | 40.95 ± 3.17     |
| LPM              | 55.35 ± 1.30     | 44.37 ± 2.81     | 77.49 ± 0.86     |

> **Note:** Table 3 reports the numerical results visualized in **Figure 4** of the paper.

#### 📊 Table 4: Impact of Auto-Labeling on Model Performance (Mean ± Std, 5-Fold Cross-Validation)

| **Model**           | **AP50 (%)**       | **AP75 (%)**       | **AP (%)**         | **F1 Score (%)**     |
|---------------------|--------------------|---------------------|---------------------|----------------------|
| Baseline            | 90.75 ± 1.25       | 83.91 ± 1.86        | 77.60 ± 0.74        | 85.14 ± 1.25         |
| Comprehensive FT    | 94.67 ± 0.74       | 90.15 ± 1.63        | 83.89 ± 1.05        | 89.44 ± 1.82         |
| **Progressive FT**  | **95.32 ± 1.27**   | **90.48 ± 2.06**    | **84.19 ± 1.37**    | **89.93 ± 1.52**     |

> **Note:** Table 4 provides detailed results corresponding to **Table II** in the paper.


## 📚 Citation

If you use this code or dataset, please cite:

```BibTeX
@inproceedings{biswas2025visualcontent,
  title     = {Visual Content Detection in Educational Videos with Transfer Learning and Dataset Enrichment},
  author    = {Biswas, Dipayan and Shah, Shishir and Subhlok, Jaspal},
  booktitle = {Proceedings of the IEEE International Conference on Multimedia Information Processing and Retrieval (MIPR)},
  year      = {2025},
  note      = {To appear}
}
```


## 📝 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
