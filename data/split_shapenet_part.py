
import numpy as np
import open3d as o3d
import os
import random
import json 
from pathlib import Path



train_test_split_dir = "shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split"
train_files_list = os.path.join(train_test_split_dir, "shuffled_train_file_list.json")
val_files_list = os.path.join(train_test_split_dir, "shuffled_val_file_list.json")
test_files_list = os.path.join(train_test_split_dir, "shuffled_test_file_list.json")

def get_train_test_list_for_category(category):
    train_list = []
    val_list = []
    test_list = []

    # Load and process train files list
    with open(train_files_list, 'r') as f:
        train_data = json.load(f)
        for item in train_data:
            if category in item:
                model_id = item.split('/')[-1]  # Extract model ID
                train_list.append(model_id)

    # Load and process validation files list
    with open(val_files_list, 'r') as f:
        val_data = json.load(f)
        for item in val_data:
            if category in item:
                model_id = item.split('/')[-1]  # Extract model ID
                val_list.append(model_id)

    # Load and process test files list
    with open(test_files_list, 'r') as f:
        test_data = json.load(f)
        for item in test_data:
            if category in item:
                model_id = item.split('/')[-1]  # Extract model ID
                test_list.append(model_id)

    return train_list, val_list, test_list

def create_shapenet_json(target_dir, train_list, val_list, test_list, category_name="airplane", taxonomy_id="02691156"):
    data = {
        "taxonomy_id": taxonomy_id,
        "taxonomy_name": category_name,
        "train": train_list,
        "val": val_list,
        "test": test_list
    }
    
    with open(os.path.join(target_dir, "ShapeNet.json"), 'w') as json_file:
        json.dump(data, json_file, indent=4)
        print("Written file to ")
        print(os.path.join(target_dir, "ShapeNet.json"))

def load_txt_as_pcd(txt_file):
    # Read the .txt file
    points = []
    with open(txt_file, 'r') as file:
        for line in file:
            # Assuming each line is: x y z nx ny nz label
            parts = line.strip().split()
            if len(parts) < 6:  # Ensure there are at least x, y, z coordinates
                continue
            # Extract only the x, y, z coordinates
            xyz = list(map(float, parts[:3]))
            points.append(xyz)
    
    # Create an Open3D point cloud from the list of points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd


def generate_pcd_files(source_dir, target_dir, model_ids, category, is_partial=False, sampling_rate=0.6):
    for model_id in model_ids:
        source_file = os.path.join(source_dir, category, model_id + ".txt")
        target_file_path = os.path.join(target_dir, 'partial' if is_partial else 'complete', category)
        Path(target_file_path).mkdir(parents=True, exist_ok=True)
        target_file = os.path.join(target_file_path, model_id + ".pcd")
        
        # Load the point cloud from .txt file
        pcd = load_txt_as_pcd(source_file)
        
        # If partial, randomly sample points
        if is_partial:
            num_points = np.asarray(pcd.points).shape[0]
            choice = np.random.choice(num_points, int(num_points * sampling_rate), replace=False)
            pcd = pcd.select_by_index(choice)
        
        # Save the point cloud in .pcd format
        o3d.io.write_point_cloud(target_file, pcd)





# create_partial_and_complete_point_clouds(input_dir, output_dir_partial, output_dir_complete, class_info_file)
target_dir = "/rhome/xaviergeorge/code/PoinTr/data/ShapeNetPart"
source_dir = "/rhome/xaviergeorge/code/PoinTr/data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
category = "02691156"  # Example category for airplanes
model_ids_train, model_ids_val, model_ids_test = get_train_test_list_for_category(category)
create_shapenet_json(target_dir, model_ids_train, model_ids_val, model_ids_test)
generate_pcd_files(source_dir, target_dir + '/train', model_ids_train, category, is_partial=False)
generate_pcd_files(source_dir, target_dir + '/train', model_ids_train, category, is_partial=True)

print("Training set done ")

generate_pcd_files(source_dir, target_dir + '/test', model_ids_test, category, is_partial=False)
generate_pcd_files(source_dir, target_dir + '/test', model_ids_test, category, is_partial=True)

print("Test set done ")

generate_pcd_files(source_dir, target_dir + '/val', model_ids_val, category, is_partial=False)
generate_pcd_files(source_dir, target_dir + '/val', model_ids_val, category, is_partial=True)

print("Val set done ")
