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

def img_transform(is_4D=True, has_inp_ch=True):
    """Implementation of callable transform function.

    Returns:
        callable: transform the input volume by changing time dim and adding channel dim.
    """
    transformation_queue = []    
    if is_4D:
        transformation_queue.append(transforms.Lambda(lambda x: x.permute(3,0,1,2)))
    if has_inp_ch:
        transformation_queue.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
    return transforms.Compose(transformation_queue)

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
        clean_up (bool, optional): if True, apply noise cleaning procedure. Bandpass filter and detrending. Defaults to False.
        smooth (bool, optional): if True, apply guassian filter on the image. Defaults to False.
        imbalanced_flag (bool, optional): True if dataset is imbalanced otherwise False. Defaults to False.
        n_timepoint (int, optional): Number of timepoints or a slice of them. Defaults to 100.
        sampling_rate (int, optional): Sampling rate of timepoints. Defaults to 1.
        stablize (bool, optional): Compute absolute value of tensor. Defaults to False.
        inp_mode (str, optional): Mode of input data which can be either fMRI or ICA. Defaults to 'fMRI'.
        data_shape (bool, optional): rearange the shape of input/output data to original shape. Defaults to True.
        valid_ids (Union[None, List[int,]], optional): Index of verified ICA components. Defaults to None.
        transform (bool, optional): If True, apply transform on input tensor. Defaults to False.
        use_peak_slice (Union[None, Tuple[int,]], optional): Returns a slice of 3D space containing the peak voxel with the given coordinate. Defaults to False.
        task (str, optional): Target task of dataset either 'Recognition' or 'DensePrediction'. Defaults to 'Recognition'.
    """   
    def __init__(self,
                 image_list_file: str,
                 dataset_name: str,
                 mask_file: str,
                 image_size: Tuple[int,...],
                 clean_up = False,
                 smooth = False,
                 imbalanced_flag = False,
                 n_timepoint = 100,
                 sampling_rate = 1,
                 stablize = False,
                 inp_mode = 'fMRI',
                 keep_shape = True,
                 has_input_channel = True,
                 valid_ids: Union[None, List[int,]] = None,
                 transform = False,
                 use_peak_slice = None,
                 task='Recognition'):
        self.info_dataframe = pd.read_pickle(image_list_file)
        self.dataset_name = dataset_name
        self.mask_img = mask_file
        self.image_size = image_size
        self.clean_noise = clean_up
        self.smoothing = smooth
        self.imbalanced_weights = compute_class_weights(self.info_dataframe.Diagnosis) if imbalanced_flag else None
        self.time_bound = n_timepoint
        self.sampling_rate = sampling_rate
        self.stablize = stablize
        self.mode = IMode(inp_mode)
        self.keep_shape = keep_shape
        self.verified_networks = valid_ids
        self.peak_slice = use_peak_slice if use_peak_slice else None
        self.transform = img_transform(is_4D=keep_shape, has_inp_ch=has_input_channel) if transform else None 
        self.class_dict = to_index(list(self.info_dataframe.Diagnosis.unique())) if task == 'Recognition' else None

    def _load_img(self, sample_idx: int) -> torch.Tensor:  
        """Load and preprocess an fMRI image.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Raises:
            ValueError: If input mode flag is set to ICA for any other dataset except BSNIP.

        Returns:
            torch.Tensor: 5D tensor of image data, (#channel_size, #timepoints, 3D space).
        """
        if (self.dataset_name !='BSNIP' and self.mode == IMode.ica):
            raise ValueError("Please change ICA input mode to fMRI since it is only valid for BSNIP.")
        if self.mode == IMode.fmri:
            img_dir = self.info_dataframe.iloc[sample_idx].FilePath
            img = fmri_preprocess(inp_img=img_dir,
                                mask_img=self.mask_img,
                                denoise=self.clean_noise,
                                blur=self.smoothing,
                                time_slice=self.time_bound,
                                step_size=self.sampling_rate,
                                rearange=self.keep_shape)
        elif self.mode == IMode.ica:
            self.verified_networks = np.arange(100) if self.verified_networks == None else self.verified_networks
            img_dir = img_dir = self.info_dataframe.iloc[sample_idx].SideInfo
            img = ica_mixture(inp_mat_file=img_dir,
                                 mask_img=self.mask_img,
                                 valid_networks=self.verified_networks,
                                 stablize=self.stablize,
                                 time_slice=self.time_bound,
                                 step_size=self.sampling_rate,
                                 mix_it=True,
                                 rearange=self.keep_shape)
        if isinstance(img, Nifti1Image):
            img = img.get_fdata()
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        return img.float()

    def _load_label_or_prior(self, sample_idx: int) -> torch.Tensor:
        """Load label/prior of each sample and convert to a tensor.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: Index of relevent class.
        """
        if self.class_dict:
            status = self.info_dataframe.iloc[sample_idx].Diagnosis
            return torch.tensor(self.class_dict[status], dtype=torch.long)
        else:
            self.verified_networks = np.arange(100) if self.verified_networks == None else self.verified_networks
            if self.dataset_name == 'BSNIP':
                img_dir = self.info_dataframe.iloc[sample_idx].SideInfo
                img = ica_mixture(img_dir,
                                    self.mask_img,
                                    self.verified_networks,
                                    self.stablize,
                                    self.time_bound,
                                    self.sampling_rate,
                                    False,
                                    self.keep_shape)
            else:
                img_dir = self.info_dataframe.iloc[sample_idx].Prior
                img = ica_mixture(img_dir,
                                    self.mask_img,
                                    self.verified_networks,
                                    self.stablize,
                                    self.time_bound,
                                    self.sampling_rate,
                                    self.keep_shape)
        if isinstance(img, Nifti1Image):
            img = img.get_fdata()
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)            
        return img.float() # type: ignore

    def __len__(self):
        return len(self.info_dataframe)

    def __getitem__(self, idx):
        img = self._load_img(idx)
        label = self._load_label_or_prior(idx)
        if self.peak_slice:
            cut_coord = slice(self.peak_slice[2]-1, self.peak_slice[2]+2)
            img = img[:,:,cut_coord]
            label = label[:,:,cut_coord]
        if self.transform:
            img = self.transform(img)
        return img, label
