"""Module contains BrainFMRIDataset class for dataloading and preprocessing."""

import torch
import scipy.io as iom
from typing import Tuple 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tools.utils import fmri_masking, tuple_prod

def _transformation():
    """Implementation of callable transform function.

    Returns:
        callable: transform the input volume
    """
    return transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

class UKBiobankDensePredictionDataset(Dataset):
    """Brain fMRI dataset with extracted ICA components as prior

    Args:
        image_dir (list): list of fMRI files' directories
        component_dir (list): list of components' directories - prior
        processing_info (list): list of meta-data files
        mask_path (str): path to mask file
        valid_components (list): indices of verified components
        component_id (int): target brain network
        min_max_scale (bool, optional): scale input volume in range [0,1]. Defaults to True.
    """    
    def __init__(self, image_dir: list, component_dir: list, processing_info: list, mask_path: str, 
                valid_components: Tuple[int,...], component_id: int, image_size: Tuple[int,...], min_max_scale=False):
        self.images = image_dir
        self.ica_maps = component_dir
        self.ica_informaion = processing_info
        self.mask_path = mask_path
        self.verified_components = valid_components
        self.ica_map_id = component_id
        self.scaling = min_max_scale
        self.image_size = image_size
        self.transform = _transformation()

    def _input_volume_load(self, subject_ind: int) -> torch.Tensor:  
        img = fmri_masking(self.images[subject_ind], self.mask_path, nor=True, sc=self.scaling)
        return torch.from_numpy(img.get_fdata()).float()

    def _ica_prior_load(self, subject_ind: int) -> torch.Tensor:    
        param = torch.Tensor(self.verified_components) - 1
        file_content = iom.loadmat(self.ica_maps[subject_ind])
        data_cube = torch.einsum('nmk,mjk->njk', torch.Tensor(file_content['ic'])[param.long(),:].T.unsqueeze(1), 
                            torch.Tensor(file_content['tc']) [:,param.long()].unsqueeze(0))
        gaussian_cdf = torch.div(torch.abs(data_cube[:,:,self.ica_map_id]), torch.abs(data_cube).sum(dim=-1)).float()
        file_content = iom.loadmat(self.ica_informaion[subject_ind])        
        vol_indices = torch.Tensor(file_content['sesInfo'][0,0]['mask_ind']).ravel() - 1
        map4d = torch.zeros(tuple_prod(self.image_size[:3]), self.image_size[-1], dtype=torch.float32)
        map4d[vol_indices.long(),:] = gaussian_cdf
        return map4d.reshape(*reversed(self.image_size[:3]), self.image_size[-1]).permute(*reversed(range(len(self.image_size[:3]))),3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self._input_volume_load(idx)
        label = self._ica_prior_load(idx)        
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        return img, label
