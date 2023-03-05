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
from tools.utils import fmri_preprocess, to_index

def img_transform():
    """Implementation of callable transform function.

    Returns:
        callable: transform the input volume by changing time dim and adding channel dim.
    """
    return transforms.Compose([
            transforms.Lambda(lambda x: x.permute(3,0,1,2).unsqueeze(0))
        ])

class ScepterViTDataset(Dataset):      
    """Dual purpose dataset class for recognition and dense prediction.

    Args:
        image_list_file (str): Path to pandas dataframe including dataset information.
        dataset_name (str): Name of dataset.
        mask_file (str): Path to mask file.
        image_size (Tuple[int,...]): Size of each fMRI image.
        min_max_scale (Union[None, Tuple[int, int]], optional): If not None, set lower and upper bound of scaling. Defaults to None.
        imbalanced_flag (bool, optional): True if dataset is imbalanced otherwise False. Defaults to False.
        n_timepoint (int, optional): Number of timepoints or a slice of them. Defaults to 100.
        normalization_dim (int, optional): If 1 timepoint-wise, 0 for voxel-wise. Defaults to 1.
        transform (bool, optional): If True, apply transform on input tensor. Defaults to False.
        task (str, optional): Target task of dataset either 'Recognition' or 'DensePrediction'. Defaults to 'Recognition'.
    """        
    def __init__(self, 
                 image_list_file: str,
                 dataset_name: str,
                 mask_file: str,
                 image_size: Tuple[int,...],
                 min_max_scale: Union[None, Tuple[int, int]] = None,
                 imbalanced_flag=False,
                 n_timepoint= 100,
                 normalization_dim: Union[None, int] = 1,
                 transform=False,
                 task='Recognition'):
        self.info_dataframe = pd.read_pickle(image_list_file)
        self.dataset_name = dataset_name
        self.mask_img = mask_file
        self.image_size = image_size
        self.scaling = min_max_scale
        self.imbalanced = imbalanced_flag
        self.time_bound = n_timepoint
        self.norm_dim = normalization_dim
        self.transform = img_transform() if transform else None 
        self.class_dict = to_index(self.info_dataframe.Diagnosis.unique()) if task == 'Recognition' else None

    def _load_img(self, sample_idx: int) -> torch.Tensor:  
        """Load and preprocess an fMRI image.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: 5D tensor of image data, (#channel_size, #timepoints, 3D space).
        """        
        img_dir = self.info_dataframe.iloc[sample_idx].FilePath
        img = fmri_preprocess(inp_img=img_dir,
                              mask_img=self.mask_img,
                              norm_dim=self.norm_dim,
                              scale=self.scaling,
                              time_slice=self.time_bound)
        return torch.from_numpy(img).float()

    def _load_label(self, sample_idx: int) -> torch.Tensor:
        """Load label of each sample and convert to one-hot encoding.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: one-hot encoding of relevent label.
        """                
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
