import torch.utils.data as data
import numpy as np
import os
import json
from pathlib import Path
import random

# Assuming data_transforms.py and io.py are properly set up in your utils package
import data_transforms
from .io import IO
from utils.logger import print_log
from .build import DATASETS

@DATASETS.register_module()
class ShapeNetPart(data.Dataset):
    def __init__(self, config):
        
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.npoints = 3072
        self.subset = config.subset
        self.category_file_path = config.CATEGORY_FILE_PATH
        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        self.segmentation_label_path = config.SEGMENTATION_LABEL_PATH
        

        with open(self.category_file_path, 'r') as f:
            self.taxonomy = json.load(f)

        self.file_list = self._get_file_list()
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded for the {self.subset} set.', logger='ShapeNetPartAirplanes')

        self.transforms = self._get_transforms()

    def _get_file_list(self):
        file_list = []
        # Assuming taxonomy information includes subset mappings
        for model_id in self.taxonomy[self.subset]:
            partial_path = os.path.join( self.partial_points_path.format(self.subset, "02691156", model_id))
            complete_path = os.path.join( self.complete_points_path.format(self.subset, "02691156", model_id))
            label_path = os.path.join(self.segmentation_label_path, "02691156", model_id + ".txt")            
            file_list.append({'model_id': model_id, 'partial_path': partial_path, 'complete_path': complete_path, 'label_path': label_path})

        return file_list
    
    def load_segmentation_labels(self, file_path):
        """
        Load segmentation labels from a file, converting them from float to int.

        Parameters:
        - file_path: Path to the segmentation label file.

        Returns:
        - labels: Numpy array of integer labels.
        """
        # Load labels as floats first
        labels_float = np.loadtxt(file_path, dtype=np.float32, usecols=-1)
        # Convert to integers
        labels_int = labels_float.astype(np.int64)
        return labels_int


    def _get_transforms(self):
        transform_list = [
            {
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': self.npoints},
                'objects': ['partial']
            }, 
            {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt', 'labels']
            }
        ]
        
        if self.subset == 'train':
            transform_list.insert(1, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            })

        return data_transforms.Compose(transform_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        partial = IO.get(sample['partial_path']).astype(np.float32)
        gt = IO.get(sample['complete_path']).astype(np.float32)

        # print("sample is ", sample['label_path'])
        labels = self.load_segmentation_labels(sample['label_path']).astype(np.int64)

        # Upsample the gt points
        # print("SElF.npoints ", self.npoints)
        gt_upsampled, labels_upsampled = self.upsample_points_and_labels(gt, labels, target_num_points=self.npoints)
        data = {'partial': partial, 'gt': gt_upsampled, 'labels': labels_upsampled}
        if self.transforms is not None:
            data = self.transforms(data)
        
        return sample['model_id'], (data['partial'], data['gt'], data['labels'])
    
    def upsample_points_and_labels(self, points, labels, target_num_points):

        current_num_points = points.shape[0]
        # print(labels.shape)
        # print(points.shape[0])
        
        # If the current number of points is already equal to or exceeds the target, return them directly
        if current_num_points >= target_num_points:
            return points[:target_num_points], labels[:target_num_points]
        
        # Calculate how many times we can repeat the existing points to approach the target number
        repeat_factor = target_num_points // current_num_points
        repeated_points = np.repeat(points, repeat_factor, axis=0)
        repeated_labels = np.repeat(labels, repeat_factor, axis=0)
        
        # Calculate how many additional points are needed to reach the target number
        additional_points_needed = target_num_points - repeated_points.shape[0]
        
        # Randomly select additional points and their corresponding labels to meet the target number
        if additional_points_needed > 0:
            additional_indices = np.random.choice(current_num_points, additional_points_needed, replace=True)
            additional_points = points[additional_indices]
            additional_labels = labels[additional_indices]
            
            # Combine the repeated points and the additional points to form the final set
            final_points = np.vstack((repeated_points, additional_points))
            final_labels = np.concatenate((repeated_labels, additional_labels))
        else:
            final_points = repeated_points
            final_labels = repeated_labels
        assert final_points.shape[0] == target_num_points
        assert final_labels.shape[0] == target_num_points
        return final_points, final_labels



    def __len__(self):
        return len(self.file_list)
