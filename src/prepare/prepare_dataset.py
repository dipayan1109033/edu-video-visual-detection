
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from collections import defaultdict
from src.prepare.dataset_utils import *
from src.utils.common import Helper



def checking_aCustom_dataset(dataset_root_path, dataset_folder, mapping=None):
    dataset_path = os.path.join(dataset_root_path, dataset_folder)

    check_bbox_coordinates(dataset_path, fixFlag=False)

    #check_images_withoutBBox(dataset_path)

    #draw_bboxes_over_images(dataset_path)

    check_objects_classes(dataset_path)
    if mapping: update_objects_classes(dataset_path, class_mappping=mapping)


def checking_dataset():
    dataset_root_path = "data/raw/"
    
    # dataset folders: sample_dataset, testdata_215, testdata_738, testdata_1k, testdata_3k3, testdata_1kcns
    # dataset folders: ldd_vdataset, lpm_dataset, lpm_vdataset

    dataset_folder = "lpm_vdataset"
    checking_aCustom_dataset(dataset_root_path, dataset_folder)






def get_image_ids(dataset_path):
    labelfolder_path = os.path.join(dataset_path, "labels")
    label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

    image_ids = []
    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)

        data = helper.read_from_json(json_filepath)
        image_id = data['asset']['image_id']
        image_ids.append(image_id)
    return image_ids

def create_datasets_image_ids_json(dataset_root_path, dataset_folders, skip_old=False):
    # Read dataset_image_ids info file (create one if not exist)
    json_filepath = os.path.join(dataset_root_path, "temp", "datasets_image_ids.json")
    if os.path.exists(json_filepath) and not skip_old:
        datasets_info = helper.read_from_json(json_filepath)
    else:
        datasets_info = {'dataset_list': [], 'max_image_id': 0}

    # Add (if not exist) or Modify image ids info for the provided datasets
    for dataset_name in dataset_folders:
        if dataset_name not in datasets_info['dataset_list']:
            datasets_info['dataset_list'].append(dataset_name)

        dataset_path = os.path.join(dataset_root_path, dataset_name)
        image_ids = get_image_ids(dataset_path)
        datasets_info[dataset_name] = {
                                        'name': dataset_name, 
                                        'start_image_id': min(image_ids), 
                                        'end_image_id': max(image_ids), 
                                        'count': len(image_ids),
                                        'update': False
                                        }

    datasets_info['max_image_id'] = max([datasets_info[dataset_folder]['end_image_id'] for dataset_folder in datasets_info['dataset_list']])
    helper.write_to_json(datasets_info, json_filepath)

def update_adataset_image_ids(dataset_root_path, dataset_folder = "sample_dataset", start_image_id=None):
    dataset_path = os.path.join(dataset_root_path, dataset_folder)

    # Read the datasets_imageids info and initialize start_image_id
    info_filepath = os.path.join(dataset_root_path, "temp", "datasets_image_ids.json")
    datasets_info = helper.read_from_json(info_filepath)
    if start_image_id is None:
        start_image_id = datasets_info['max_image_id'] + 1

    # Check if it is allowed to update the image_ids for the provided dataset
    if dataset_folder not in datasets_info['dataset_list']:
        datasets_info['dataset_list'].append(dataset_folder)
    else:
        if not datasets_info[dataset_folder]['update']:
            print(f"Dataset '{dataset_folder}' already exist in datasets_image_ids info file and it's update is restricted")
            return

    # Actual update of image ids
    labelfolder_path = os.path.join(dataset_path, "labels")
    label_filelist = sorted(helper.get_files_with_extension(labelfolder_path, extension=".json"))
    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)

        data = helper.read_from_json(json_filepath)
        data['asset']['image_id'] = start_image_id
        start_image_id += 1
        helper.write_to_json(data, json_filepath)

    # Update the datasets_imageids info json file
    image_ids = get_image_ids(dataset_path)
    datasets_info[dataset_folder] = {
                                    'name': dataset_folder, 
                                    'start_image_id': min(image_ids), 
                                    'end_image_id': max(image_ids), 
                                    'count': len(image_ids),
                                    'update': False
                                    }
    datasets_info['max_image_id'] = max([datasets_info[dataset_folder]['end_image_id'] for dataset_folder in datasets_info['dataset_list']])
    helper.write_to_json(datasets_info, info_filepath)
    print(f"Image ids update to '{dataset_folder}' is successfull.")
    
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


def monitor_datasets_info():
    dataset_root_path = "data/raw/"

    # dataset folders: sample_dataset, testdata_215, testdata_738, testdata_1k, testdata_3k3, testdata_1kcns
    # dataset folders: ldd_dataset, lpm_dataset, lpm_vdataset


    dataset_folders = ['ldd_dataset', 'testdata_215']
    #create_datasets_image_ids_json(dataset_root_path, dataset_folders)

    
    dataset_folder = "testdata_1kcns"
    #update_adataset_image_ids(dataset_root_path, dataset_folder)

    for dataset_folder in ["ldd_vdataset", "lpm_dataset", "testdata_1kcns", "testdata_3k3cns"]:
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






# For multiple classes detection
label_dict = {"Table": 1, "Chart-Graph": 2, "Photographic-image": 3, "Visual-illustration": 4}


def prepare_dataset():
    dataset_root_path = "data/raw/"
    output_root_dir = "data/processed/"

    # dataset folders: sample_dataset, testdata_215, testdata_738, testdata_1k, testdata_3k3, testdata_1kcns, testdata_3k3cns
    # dataset folders: ldd_vdataset, lpm_dataset, lpm_vdataset, td4k_auto0.4, td4k_auto0.5

    dataset_path = os.path.join(dataset_root_path, "testdata_1kcns")
    output_folder = "testdata_1kcns_4classes"
    #check_objects_classes(dataset_path)

    label_dict = {"Table": 1, "Chart-Graph": 2, "Photographic-image": 3, "Visual-illustration": 4}
    process_dataset(dataset_path, output_root_dir, output_folder, singleClass=False, label_dict=label_dict)

    dataset_folder = "td4k_auto0.5"
    #create_dataset_info(output_root_dir, dataset_folder, singleClass=True)

    # dataset_path = os.path.join(output_root_dir, output_folder)
    # check_objects_classes(dataset_path)


def filter_labels_by_score(input_folder, output_folder, threshold=0.5):
    # output labels folder
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
    dataset_root_path = "data/raw/testdata_3k3cns"
    
    #check_objects_classes(dataset_path)

    #input_folder = os.path.join(dataset_path, "labels")
    #filter_labels_by_score(input_folder, output_folder="labels2", threshold=0.25)
    create_score_thresholded_dataset(dataset_path, label_folder="labels28", output_folder="thresholdedDataset28_0.5Up", lThreshold=0.50, uThreshold=1.0)

    dataset_path = os.path.join(dataset_path, "thresholdedDataset_0.5Up")
    #draw_bboxes_over_images(dataset_path)






if __name__ == "__main__":
    helper = Helper()

    # Check and update for inconsistency within the dataset
    #checking_dataset()

    # Monitor and update datasets info in relation to other datasets 
    monitor_datasets_info()

    # Prepare dataset for experiments
    #prepare_dataset()


    dataset_path = "data/raw/testdata_3k3cns"
    #prepare_autolabel_dataset(dataset_path)
