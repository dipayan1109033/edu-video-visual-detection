
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from collections import defaultdict
from src.prepare.dataset_utils import *
from src.utils.common import Helper




def checking_custom_dataset(dataset_root_path, dataset_folder):
    dataset_path = os.path.join(dataset_root_path, dataset_folder)

    check_bbox_coordinates(dataset_path, fixFlag=False)

    check_images_withoutBBox(dataset_path)

    draw_bboxes_over_images(dataset_path)

    check_objects_classes(dataset_path)




def print_dataset_stats(dataset_path):
    labels_folder_path = os.path.join(dataset_path, "labels")
    json_files = helper.get_files_with_extension(labels_folder_path, extension="json")

    total_objects = 0
    zero_annotation = 0
    obj_dict = defaultdict(int)
    objects_category_counts = {}

    for json_file in json_files:
        json_file_path = os.path.join(labels_folder_path, json_file)
        data = helper.read_from_json(json_file_path)

        objects_count = len(data["objects"])
        if objects_count == 0:
            zero_annotation += 1
        else:
            total_objects += objects_count
            obj_dict[objects_count] += 1

            for object in data["objects"]:
                category_name = object["class"]
                if category_name not in objects_category_counts:
                    objects_category_counts[category_name] = 1
                else:
                    objects_category_counts[category_name] += 1

    obj_dict = dict(sorted(obj_dict.items(), key=lambda x: (isinstance(x[0], int), x[0])))

    print(f"	Total images: {len(json_files)}")
    print(f"	Total objects: {total_objects}\n")
    print(f"	Image count without objects: {zero_annotation}")

    # Compute percentages for 1, 2, 3, 4, 5, more than 5 objects separately
    num_objects = 4
    distribution = {i: (obj_dict.get(i, 0) / len(json_files)) * 100 for i in range(1, num_objects+1)}
    count_above_5 = sum(v for k, v in obj_dict.items() if k > num_objects)
    distribution[f">{num_objects} objects"] = (count_above_5 / len(json_files)) * 100
    distribution = {k: round(v, 3) for k, v in distribution.items()}    # Round values for readability
    print(f"        Raw object_dict: {obj_dict}")
    print(f"	Object distribution: {distribution}")

    objects_category_counts = dict(sorted(objects_category_counts.items(), key=lambda item: item[1], reverse=True))
    print(f"	{objects_category_counts}")

def print_datasets_info():
    # LVVO_1k_withCategories, LVVO_1k, LVVO_3k, ldd_dataset, lpm_dataset
    for dataset_folder in ["LVVO_1k_withCategories", "LVVO_1k", "LVVO_3k", "ldd_dataset", "lpm_dataset"]:
        print("\nFor dataset:", dataset_folder)
        dataset_path = os.path.join(dataset_root_path, dataset_folder)
        print_dataset_stats(dataset_path)
        #break




def process_dataset(dataset_path, output_root_dir, output_folder=None, singleClass=True, label_dict=None):
    if not singleClass and not label_dict:
        raise ValueError(f"The 'label_dict' can't be None for multiple classes.")
    # Setup output folders
    if output_folder is None:
        output_folder = helper.get_immediate_folder_name(dataset_path)
    output_path = os.path.join(output_root_dir, output_folder)
    output_images_path, output_labels_path = helper.create_subfolders(output_path)

    # Get input folders
    imagefolder_path = os.path.join(dataset_path, "images")
    labelfolder_path = os.path.join(dataset_path, "labels") 

    # transfer all images and labels
    helper.copy_files_withExtensions(imagefolder_path, output_images_path, extensions=['.jpg', '.png'])
    helper.copy_files_withExtension(labelfolder_path, output_labels_path, extension='.json')

    # Convert Class Name to Class IDs
    json_files = helper.get_files_with_extension(output_labels_path, extension=".json")
    for json_file in json_files:
        json_filepath = os.path.join(output_labels_path, json_file)
        data = helper.read_from_json(json_filepath)
        for obj in data['objects']:
            if singleClass: 
                class_id = 1
            else: 
                class_id = label_dict[obj['class']]
            obj['class'] = class_id
        helper.write_to_json(data, json_filepath)

    create_dataset_info(output_root_dir, output_folder, singleClass=singleClass, label_dict=label_dict)

def prepare_dataset():
    input_root_dir = "data/raw/"

    # Covert raw dataset class names to class IDs and save into the processed folder
    dataset_path = os.path.join(input_root_dir, "LVVO_1k_withCategories")
    label_dict = {"Table": 1, "Chart-Graph": 2, "Photographic-image": 3, "Visual-illustration": 4}

    output_folder = "LVVO_1k_withCategories"
    process_dataset(dataset_path, dataset_root_path, output_folder, singleClass=False, label_dict=label_dict)


    # # Create dataset info file for the processed dataset
    # dataset_folder = "LVVO_1k_withCategories"
    # create_dataset_info(dataset_root_path, dataset_folder, singleClass=True)




def filter_labels_by_score(input_folder, output_folder, threshold=0.5):
    # output labels folder
    dataset_path = helper.get_immediate_folder_path(input_folder)
    output_labels_path = os.path.join(dataset_path, output_folder)
    os.makedirs(output_labels_path, exist_ok=True)

    label_files = helper.get_files_with_extension(input_folder, extension=".json")
    for label_file in label_files:
        label_filepath = os.path.join(input_folder, label_file)
        data = helper.read_from_json(label_filepath)

        box_list = []
        for box in data['objects']:
            box['class'] = 1
            if box['score'] >= threshold:
                box_list.append(box)

        # Create new label file
        new_label_filepath = os.path.join(output_labels_path, label_file)
        data['objects'] = box_list
        helper.write_to_json(data, new_label_filepath)

def create_score_thresholded_dataset(dataset_path, label_folder, output_folder, lThreshold=0.5, uThreshold=1.0, copy_images=True):
    # input folders
    input_image_folder, input_label_folder = helper.get_subfolder_paths(dataset_path, folder_list=["images", label_folder])

    # output labels folder
    output_path = os.path.join(dataset_path, output_folder)
    output_image_folder, output_label_folder = helper.create_subfolders(output_path)

    # filter labels by score
    label_files = helper.get_files_with_extension(input_label_folder, extension=".json")
    for label_file in label_files:
        label_filepath = os.path.join(input_label_folder, label_file)
        data = helper.read_from_json(label_filepath)

        box_list = []
        for box in data['objects']:
            
            box['class'] = 1
            if lThreshold <= box['score'] <= uThreshold:
                box_list.append(box)
                #print(box['score'])

        # Create new label file
        #if len(box_list)>0:
        new_label_filepath = os.path.join(output_label_folder, label_file)
        data['objects'] = box_list
        helper.write_to_json(data, new_label_filepath)

        if copy_images:
            image_file = label_file.replace(".json", ".jpg")
            image_path = os.path.join(input_image_folder, image_file)
            shutil.copy(image_path, output_image_folder)

def prepare_autolabel_dataset(dataset_path):
    
    # Filter predicted labels by score and create a new dataset
    input_labels_folder = os.path.join(dataset_path, "labels")
    #filter_labels_by_score(input_labels_folder, output_folder="labels2", threshold=0.50)

    output_folder="ThDataset2_0.5Up"
    create_score_thresholded_dataset(dataset_path, label_folder="labels2", output_folder=output_folder, lThreshold=0.50, uThreshold=1.0)

    dataset_path = os.path.join(dataset_path, output_folder)
    draw_bboxes_over_images(dataset_path)




if __name__ == "__main__":
    helper = Helper()
    dataset_root_path = "data/processed"
    # dataset folders: LVVO_1k_withCategories, LVVO_1k, LVVO_3k, ldd_vdataset, lpm_dataset


    # Check and update for inconsistency within the dataset
    dataset_folder = "LVVO_1k_withCategories"
    checking_custom_dataset(dataset_root_path, dataset_folder)

    # # Print datasets stats 
    # print_datasets_info()

    # # Prepare dataset for experiments
    # prepare_dataset()


    # dataset_path = "data/raw/LVVO_3k"
    # prepare_autolabel_dataset(dataset_path)
