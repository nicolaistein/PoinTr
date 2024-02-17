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
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.category_file_path = config.CATEGORY_FILE_PATH
        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        

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
            label_path = self.segmentation_label_path.format(self.subset, "02691156", model_id)  # Path for the segmentation label
            file_list.append({'model_id': model_id, 'partial_path': partial_path, 'complete_path': complete_path, 'label_path': label_path})

        return file_list
    
    def load_segmentation_labels(file_path):
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
                'objects': ['partial', 'gt']
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
        labels = load_segmentation_labels(sample['label_path'])

        # Upsample the gt points
        gt_upsampled, labels_upsampled = self.upsample_points_and_labels(gt, labels, target_num_points=self.npoints)
        data = {'partial': partial, 'gt': gt, 'labels': labels}
        if self.transforms is not None:
            data = self.transforms(data)

        return sample['model_id'], (data['partial'], data['gt'])
    
    def upsample_points_and_labels(self, points, labels, target_num_points):
        current_num_points = points.shape[0]
        if current_num_points >= target_num_points:
            return points, labels
        
        repeat_factor = target_num_points // current_num_points
        additional_points = target_num_points - (repeat_factor * current_num_points)
        
        upsampled_points = np.repeat(points, repeat_factor, axis=0)
        upsampled_labels = np.repeat(labels, repeat_factor, axis=0)
        
        if additional_points > 0:
            additional_indices = np.random.choice(current_num_points, additional_points, replace=True)
            additional_points = points[additional_indices]
            additional_labels = labels[additional_indices]  # Duplicate labels for the additional points
            
            upsampled_points = np.vstack((upsampled_points, additional_points))
            upsampled_labels = np.concatenate((upsampled_labels, additional_labels))
        
        return upsampled_points, upsampled_labels



    def __len__(self):
        return len(self.file_list)
