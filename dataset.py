import os
import numpy as np
import torch
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class TwoVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, num_volumes=2, transform=None):
        self.root_folder_path = root_folder_path
        self.num_volumes = num_volumes
        self.transform = transform
        self.preloaded_data = self.preload_volumes(root_folder_path)
        self.pairs = self.create_pairs()

    def preload_volumes(self, root_folder_path):
        preloaded_data = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            if self.num_volumes:
                sorted_files = sorted_files[:self.num_volumes]
                print(sorted_files)
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                preloaded_data[full_path] = volume
        return preloaded_data

    def create_pairs(self):
        pairs = []
        volume_paths = sorted(self.preloaded_data.keys())
        num_volumes = len(volume_paths)

        if num_volumes < 2:
            raise ValueError("There must be at least two volumes in the folder")

        for i in range(num_volumes - 1):
            input_volume_path = volume_paths[i]
            target_volume_path = volume_paths[i + 1]
            num_slices = self.preloaded_data[input_volume_path].shape[0]
            for slice_index in range(num_slices):
                pairs.append((input_volume_path, target_volume_path, slice_index))
        
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_volume_path, target_volume_path, slice_index = self.pairs[index]

        input_slice = self.preloaded_data[input_volume_path][slice_index]
        target_slice = self.preloaded_data[target_volume_path][slice_index]

        if self.transform:
            input_slice, target_slice = self.transform((input_slice, target_slice))

        input_slice = input_slice[np.newaxis, ...]
        target_slice = target_slice[np.newaxis, ...]

        return input_slice, target_slice

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.slices = self.preload_and_make_slices(root_folder_path)

    def preload_and_make_slices(self, root_folder_path):
        slices = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                for i in range(num_slices):  # Include all slices
                    slices.append((full_path, i))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        file_path, slice_index = self.slices[index]
        
        # Access preloaded data instead of reading from file
        input_slice = self.preloaded_data[file_path][slice_index]

        if self.transform:
            input_slice = self.transform(input_slice)
        
        # Add extra channel axis at position 0
        input_slice = input_slice[np.newaxis, ...]

        return input_slice