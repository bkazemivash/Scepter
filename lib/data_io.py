"""
Dataset class for brain scans - fMRI data for spatiotemporal 
classification of brain disease.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from typing import Tuple, Union 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from ..tools.utils import fmri_preprocess, to_index

def img_transform():
    """Implementation of callable transform function.

    Returns:
        callable: transform the input volume by changing time dim and adding channel dim
    """
    return transforms.Compose([
            transforms.Lambda(lambda x: x.permute(3,0,1,2).unsqueeze(0))
        ])

class ScepterViTDataset(Dataset):      
    def __init__(self, 
                 image_list_file: str,
                 dataset_name: str,
                 mask_file: str,
                 image_size: Tuple[int,...],
                 min_max_scale: Union[None, Tuple[int, int]] = None,
                 imbalanced_flag=False,
                 n_timepoint= 100,
                 normalization_dim= 1,
                 transform=False,
                 Task='Classification'):
        self.info_dataframe = pd.read_pickle(image_list_file)
        self.dataset_name = dataset_name
        self.mask_img = mask_file
        self.image_size = image_size
        self.scaling = min_max_scale
        self.imbalanced = imbalanced_flag
        self.time_bound = n_timepoint
        self.norm_dim = normalization_dim
        self.transform = img_transform() if transform else None 
        self.class_dict = to_index(self.info_dataframe.Diagnosis.unique()) if Task == 'Classification' else None

    def _load_img(self, sample_idx: int) -> torch.Tensor:  
        img_dir = self.info_dataframe.iloc[sample_idx].FilePath
        img = fmri_preprocess(inp_img=img_dir,
                              mask_img=self.mask_img,
                              ax=self.norm_dim,
                              normalize=True,
                              scale=self.scaling,
                              time_slice=self.time_bound)
        return torch.from_numpy(img).float()

    def _load_label(self, sample_idx: int) -> torch.Tensor:    
        assert self.class_dict !=  None, ValueError('Class labels are not defined!')
        status = self.info_dataframe.iloc[sample_idx].Diagnosis
        status_idx = torch.tensor(self.class_dict[status], dtype=torch.int64)
        return F.one_hot(status_idx, num_classes=len(self.class_dict))

    def __len__(self):
        return len(self.info_dataframe)

    def __getitem__(self, idx):
        img = self._load_img(idx)
        label = self._load_label(idx)        
        if self.transform:
            img = self.transform(img)
        return img, label
