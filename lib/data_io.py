"""
Dataset class for brain scans - fMRI data for spatiotemporal 
classification of brain disease.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from enum import Enum
from typing import Tuple, Union 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tools.utils import *

def img_transform():
    """Implementation of callable transform function.

    Returns:
        callable: transform the input volume by changing time dim and adding channel dim.
    """
    return transforms.Compose([
            transforms.Lambda(lambda x: x.permute(3,0,1,2).unsqueeze(0))
        ])

class IMode(Enum):
    """ An enumeration representing the possible input modes for the model."""

    fmri = 'fMRI'
    ica = 'ICA'

class ScepterViTDataset(Dataset):      
    """Dual purpose dataset class for recognition and dense prediction.

    Args:
        image_list_file (str): Path to pandas dataframe including dataset information.
        dataset_name (str): Name of dataset.
        mask_file (str): Path to mask file.
        image_size (Tuple[int,...]): Size of each fMRI image.
        min_max_scale (Union[None, Tuple[int, int]], optional): If not None, set lower and upper bound of scaling. Defaults to None.
        clean_up (bool): if True, apply noise cleaning procedure. Bandpass filter and detrending. Defaults to False.
        smooth (bool): if True, apply guassian filter on the image. Defaults to False.
        imbalanced_flag (bool, optional): True if dataset is imbalanced otherwise False. Defaults to False.
        n_timepoint (int, optional): Number of timepoints or a slice of them. Defaults to 100.
        sampling_rate (int, optional): Sampling rate of timepoints. Defaults to 1.
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
                 clean_up = False,
                 smooth = False,
                 imbalanced_flag = False,
                 n_timepoint = 100,
                 sampling_rate = 1,
                 normalization_dim: Union[None, int] = 1,
                 inp_mode = 'fMRI',
                 valid_ids: Union[None, List[int,]] = None,
                 transform = False,
                 task='Recognition'):
        self.info_dataframe = pd.read_pickle(image_list_file)
        self.dataset_name = dataset_name
        self.mask_img = mask_file
        self.image_size = image_size
        self.scaling = min_max_scale
        self.clean_noise = clean_up
        self.smoothing = smooth
        self.imbalanced_weights = compute_class_weights(self.info_dataframe.Diagnosis) if imbalanced_flag else None
        self.time_bound = n_timepoint
        self.sampling_rate = sampling_rate
        self.norm_dim = normalization_dim
        self.mode = IMode(inp_mode)
        self.verified_networks = valid_ids
        self.transform = img_transform() if transform else None 
        self.class_dict = to_index(list(self.info_dataframe.Diagnosis.unique())) if task == 'Recognition' else None

    def _load_img(self, sample_idx: int) -> torch.Tensor:  
        """Load and preprocess an fMRI image.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: 5D tensor of image data, (#channel_size, #timepoints, 3D space).
        """
        if self.mode == IMode.fmri:
            img_dir = self.info_dataframe.iloc[sample_idx].FilePath
            img = fmri_preprocess(inp_img=img_dir,
                                mask_img=self.mask_img,
                                denoise=self.clean_noise,
                                blur=self.smoothing,
                                norm_dim=self.norm_dim,
                                scale=self.scaling,
                                time_slice=self.time_bound,
                                step_size=self.sampling_rate)
        elif self.mode == IMode.ica:
            self.verified_networks = np.arange(100) if self.verified_networks == None else self.verified_networks
            img_dir = img_dir = self.info_dataframe.iloc[sample_idx].SideInfo
            img = ica_preprocess(inp_mat_file=img_dir,
                                 mask_img=self.mask_img,
                                 valid_networks=self.verified_networks,
                                 norm_dim=self.norm_dim,
                                 time_slice=self.time_bound,
                                 step_size=self.sampling_rate)

        proc_data = img.get_fdata()
        return torch.from_numpy(proc_data).float()

    def _load_label(self, sample_idx: int) -> torch.Tensor:
        """Load label of each sample and convert to a tensor.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: Index of relevent class.
        """                
        assert self.class_dict !=  None, ValueError('Class labels are not defined!')
        status = self.info_dataframe.iloc[sample_idx].Diagnosis
        return torch.tensor(self.class_dict[status], dtype=torch.long)

    def __len__(self):
        return len(self.info_dataframe)

    def __getitem__(self, idx):
        img = self._load_img(idx)
        label = self._load_label(idx)        
        if self.transform:
            img = self.transform(img)
        return img, label
