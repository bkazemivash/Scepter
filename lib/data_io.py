"""
Dataset class for brain scans - fMRI data for spatiotemporal 
classification of brain disease.
"""

import torch
import pandas as pd
from typing import Tuple 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tools.utils import fmri_preprocess, tuple_prod

def img_transform():
    """Implementation of callable transform function.

    Returns:
        callable: transform the input volume
    """
    return transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

class ScepterViTDataset(Dataset):  
    def __init__(self, 
                 image_list_file: str,
                 dataset_name: str,
                 mask_file: str,
                 image_size: Tuple[int,...],
                 min_max_scale=False,
                 imbalanced_flag=False,
                 n_timepoint= 100,
                 transform=False,
                 Task='Classification'):
        self.info_dataframe = pd.read_pickle(image_list_file)
        self.dataset_name = dataset_name
        self.mask_img = mask_file
        self.image_size = image_size
        self.scaling = min_max_scale
        self.imbalanced = imbalanced_flag
        self.time_bound = n_timepoint
        self.transform = img_transform() if transform else None 
        self.classes = None if Task == 'DensePrediction' else self.info_dataframe.Diagnosis.unique() 

    def _load_img(self, sample_idx: int) -> torch.Tensor:  
        img_dir = self.info_dataframe[sample_idx].FilePath
        img = fmri_preprocess(inp_img=img_dir,
                              mask_img=self.mask_img,
                              ax=1,
                              normalize=True,
                              scale=(-2, 2),
                              time_slice=100)
        return torch.from_numpy(img).float()

    def _load_label(self, sample_idx: int) -> torch.Tensor:    
        status = self.info_dataframe[sample_idx].Diagnosis
        

        





    def __len__(self):
        return len(self.info_dataframe)

    def __getitem__(self, idx):
        img = self._load_img(idx)
        label = self._load_label(idx)        
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        return img, label
