"""
Helper functions for verification and preprocessing.
This module contains mandatory functions to run different preprocessing and 
postprocessing steps on input fMRI images.
"""

import os
import numpy as np
import operator
import torch
import torch.nn as nn
import mat73
from multipledispatch import dispatch
from omegaconf import ListConfig
from typing import Tuple, Union, Dict, Any, List
from nilearn.masking import unmask, apply_mask
from nilearn.image import index_img, load_img
from nibabel.nifti1 import Nifti1Image
from scipy import signal
from scipy.ndimage import gaussian_filter 
from functools import reduce
from math import floor


def weights_init(m: nn.Module) -> None:
    """ Initialize model weights using xavier method

    Args:
        m (nn.Module): model layers
    """
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.xavier_normal_(m.weight.data, gain=.75)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.Conv2d,)):
        nn.init.xavier_normal_(m.weight.data, gain=.75)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)            
    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)     

   
def tuple_prod(inp: Tuple[int, ...]) -> int:
    """ Multiplies all elements of the input tuple.
       -- Equivalent to numpy.prod, but with lower running time in our case
       
    Args:
        inp (Tuple[int, ...]): input tuple

    Returns:
        int: multiplication of all elements
    """    
    return reduce(operator.mul, inp, 1)


def configure_patch_embedding(inp_dim: Tuple[int, ...], patch_size: int, 
                              down_sampling: float,) ->Tuple[Tuple[int, ...], int]:
    """Computes number of patches and decoder head dimension

    Args:
        inp_dim (Tuple[int, ...]): Size of input tensor
        patch_size (int): Size of the patch
        down_sampling (float): Ratio of down sampling

    Returns:
        Tuple[Tuple[int, ...], int]: Includes dimension of decoder head and number of patches in totall.
    """
    if down_sampling < 1.:
        inp_dim = tuple(floor(i*down_sampling) if floor(i*down_sampling) > 3 else 3 for i in inp_dim)
    decoder_head = tuple(map(lambda x: x//patch_size, inp_dim))
    patch_num = reduce(operator.mul, decoder_head, 1)
    return (decoder_head, patch_num, inp_dim)


def get_num_patches(inp_dim: Tuple[int, ...], patch_size: int) -> int:
    """Computes totall number of patches

    Args:
        inp_dim (Tuple[int, ...]): Size of input tensor
        patch_size (int): Size of each patch

    Returns:
        int: Totall number of patches
    """
    return reduce(operator.mul, tuple(map(lambda x: x//patch_size, inp_dim)), 1)


def scale_array(ar: np.ndarray, lb = 0, ub = 1, ax = -1) -> np.ndarray:
    """ Scale input array in range of [lb, ub]

    Args:
        ar (np.ndarray): input array to be scaled
        lb (int, optional): lower bound of scaling function. Defaults to 0.
        ub (int, optional): upper bound of scaling function. Defaults to 1.        
        ax (int, optional): axis to apply scaling function. Defaults to -1.

    Returns:
        np.ndarray: scaled array ranging in [lb, ub]
    """
    return lb + ((ub - lb) * (np.subtract(ar, np.min(ar, axis=ax, keepdims=True))) / np.ptp(ar, axis=ax, keepdims=True))


def butter_bandpass_filter(ar: np.ndarray, 
                           cut_off_threshold: List[Union[None, float]] = [.02, .15], 
                           sampling_rate=.5, 
                           order = 5) -> np.ndarray:
    b, a = signal.butter(order, cut_off_threshold, fs=sampling_rate, btype='band')
    padding = ar.shape[0] - 1
    filtered_signal = signal.filtfilt(b, a, ar, axis=0, padlen=padding)
    return filtered_signal


def fmri_preprocess(inp_img: Union[str, Nifti1Image],
                    mask_img: str,
                    denoise: bool = False,
                    blur: bool = False,
                    time_slice =0,
                    step_size=1,
                    rearange = True) -> Union[Nifti1Image, np.ndarray]:
    """ Mask, z-score, and scale input fMRI image.

    Args:
        inp_img (Union[str, Nifti1Image]): A 4D Niimg-like object or a directory of it.
        mask_img (str): Path to a 3D Niimg-like mask object.
        denoise (bool): If True, performs denoising procedure for fMRI data. Defaults to False.
        blur (bool): If True, perform gaussian filter on the image. Defaults to False.
        time_slice (int, optional): Slice of timepoints. Defaults to 0.
        step_size (int, optional): Sampling rate of timepoints. Defaults to 1.
        rearrange (bool, optional): Rearranging the output matrix to original 4G. Defaults to True.

    Raises:
        TypeError: If 'inp_img' is not a Niimg-like object.

    Returns:
        Union[Nifti1Image, np.ndarray]: a 4D Niimg-like object or ndarray of shape #timepoints by #voxel.
    """    
    if ((not isinstance(inp_img, Nifti1Image)) and (isinstance(inp_img, str) and not (inp_img.lower().endswith(('.nii', '.nii.gz'))))):
        raise TypeError("Input image is not a Nifti file, please check your input!") 

    totall_timepoints = time_slice * step_size
    original_img = inp_img
    if time_slice > 0:
        original_img = index_img(inp_img, slice(0, totall_timepoints, step_size)) # type: ignore    
    data_ = apply_mask(original_img, mask_img)
    if blur:
        data_ = gaussian_filter(data_, sigma=0.5)
    data_ = (data_ - data_.mean()) / data_.std()
    if denoise:
        cutoff = [data_.mean() - (3. * data_.std()), data_.mean() + (3. * data_.std())]
        data_[(data_>cutoff[1]) | (data_<cutoff[0])] = 0.
        data_[data_ != 0.] = 5.999 * (data_[data_ != 0.] - data_[data_ != 0.].min()) / (data_[data_ != 0.].max() - data_[data_ != 0.].min()) + .001
    if rearange:
        data_ = unmask(data_, mask_img)
    return data_  # type: ignore


def path_correction(inp_path: str) -> tuple[str, str]:
    """Generating paths for spatial/temporal nii files.

    Args:
        inp_path (str): Path to the ICA mat file.

    Returns:
        tuple[str, str]: Paths to spatial maps and time-courses files.
    """    
    splited_path = inp_path.replace('.mat', 'component').split('component')
    spatial_mat_data = ''.join([splited_path[0], 'component', splited_path[1], '.nii'])
    temporal_mat_data = ''.join([splited_path[0], 'timecourses', splited_path[1], '.nii'])
    return spatial_mat_data, temporal_mat_data


@dispatch(str, str, ListConfig, bool, int, int, bool)
def ica_mixture(inp_mat_file: str,
                   mask_img: str,
                   valid_networks: List[int,],
                   stablize = False,
                   time_slice = 0,
                   step_size = 1,
                   rearange = False) ->  torch.Tensor:
    """Load and preprocess spatially constrined windowed ICA, stored in pt files.
        Note: We don't z-score here as data is already normal.

    Args:
        inp_mat_file (str): Path to pt file including detail information.
        mask_img (str): Path to a 3D Niimg-like mask object.
        valid_networks (List[int,]): List of all verified ICA components.
        stablize (bool, optional): Get absolute value of data. Dafaults to False.
        time_slice (int, optional): Slice of timepoints. Defaults to 0.
        step_size (int, optional): Sampling rate of timepoints. Defaults to 1.
        rearrange (bool, optional): Rearranging the output matrix to original 4G. Defaults to False.

    Returns:
        torch.Tensor: torch Tensor including 4D maps.
    """
    valid_idx = np.array(valid_networks) - 1
    steper = slice(0,None)
    if time_slice>0:
        totall_timepoints = time_slice * step_size
        steper = slice(0, totall_timepoints, step_size)
    mask_data = load_img(mask_img).get_fdata()
    mask_shape_ = mask_data.shape
    data_ = torch.load(inp_mat_file)[:, valid_idx, steper].permute(0,2,1)
    if stablize:
        data_ = 2 * (data_ - data_.amin(dim=(0,1)))/(data_.amax(dim=(0,1)) - data_.amin(dim=(0,1))) -1
    if rearange:
        mask_data = mask_data.flatten(order='F').astype('bool')
        mask_data = torch.from_numpy(mask_data)
        reshaped_data = torch.zeros(mask_data.shape + (data_.shape[1],), dtype=torch.double)
        reshaped_data[mask_data == True] = data_.squeeze()
        data_ = reshaped_data.reshape(*reversed(mask_shape_), data_.shape[1]).permute(2,1,0,3)
    return data_


@dispatch(str, str, ListConfig, bool, int, int, bool, bool)
def ica_mixture(inp_mat_file: str,
                   mask_img: str,
                   valid_networks: List[int,],
                   stablize = False,
                   time_slice = 0,
                   step_size = 1,
                   mix_it = False,
                   rearange = False) -> Union[Nifti1Image, torch.Tensor]:
    """Preprocess ICA time-courses and spatial maps.

    Args:
        inp_mat_file (str): Path to mat file including detail information.
        mask_img (str): Path to a 3D Niimg-like mask object.
        valid_networks (List[int,]): List of all verified ICA components.
        stablize (bool, optional): Get absolute value of data. Dafaults to False.
        time_slice (int, optional): Slice of timepoints. Defaults to 0.
        step_size (int, optional): Sampling rate of timepoints. Defaults to 1.
        mix_it (bool, optional): generating mixture of all components/timepoits. Defaults to False.
        rearrange (bool, optional): Rearranging the output matrix to original 4G. Defaults to False.

    Returns:
        Union[Nifti1Image, torch.Tensor]: a 4D Niimg-like object or processed ICA components.
    """
    valid_idx = np.array(valid_networks) - 1
    spatial_map, time_course = path_correction(inp_mat_file)
    spatial_data = torch.from_numpy(apply_mask(spatial_map, mask_img))
    temporal_data = torch.from_numpy(load_img(time_course).get_fdata()) # type: ignore
    steper = slice(0,None)
    if time_slice>0:
        totall_timepoints = time_slice * step_size
        steper = slice(0, totall_timepoints, step_size)
    command_ = 'bp,pq->bq' if mix_it else 'bp,pq->bqp'
    data_ = torch.einsum(command_, temporal_data[steper, valid_idx], spatial_data[valid_idx,:])
    data_ = data_ - data_.mean(dim=(0,1)) / data_.std(dim=(0,1))
    if stablize:
        data_ = data_.abs()
        data_ = 5.999 * (data_ - data_.amin(dim=(0,1)))/(data_.amax(dim=(0,1)) - data_.amin(dim=(0,1))) + 0.001
    if rearange:
        data_ = data_.squeeze()
        data_ = unmask(data_, mask_img)
    return data_   # type: ignore


def get_n_params(model: nn.Module) -> int:
    """ Gets the number of model parameters.

    Args:
        model (nn.Module): An nn.module object.

    Returns:
        int: Totall number of parameters.
    """    
    return sum(p.numel() for p in model.parameters())


def assert_tensors_equal(t1: torch.Tensor, t2: torch.Tensor) -> None:
    """ Sanity check: Raises an AssertionError if two tensors
        are not equal.

    Args:
        t1 (Tensor): first given tensor.
        t2 (Tensor): second given tensor. 
    """    
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)


def get_key_by_value(dict_: Dict, val: int) -> str:
    """Get the original class label from class dictionary

    Args:
        dict_ (Dict): Input class dictionary
        val (int): Index of class

    Returns:
        str: Relevant class label
    """    
    key_ = list(filter(lambda x: dict_[x] == val, dict_))[0]
    return key_


def to_index(class_labels: List) -> Dict:
    """Make a dict with keys:labels to value: index

    Args:
        class_labels (np.ndarray): list of class labels

    Returns:
        Dict: key, value : label, index
    """    
    class_labels = sorted(class_labels)
    label_index_dict = {label: index for index, label in enumerate(class_labels)}
    return label_index_dict


def compute_class_weights(x: Any) -> torch.Tensor:
    """Compute class weights in an imbalanced dataset

    Args:
        x (Any): Array-like object including classes 

    Returns:
        torch.Tensor: Class weight tensor
    """    
    _, class_count = np.unique(x, return_counts=True)
    weights = 1. / class_count
    return torch.tensor(weights, dtype=torch.float)


def memory_report() -> str:
    """Checks the memory status on the gpu node.

    Returns:
        str: Reports different elements of memory.
    """
    report = "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/ 1024**3) +\
          "\ttorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/ 1024**3) +\
          "\ttorch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/ 1024**3)

    return report


def fetch_list_of_backup_files(model_arch: str) -> tuple:
    """Defines a list of files to backup based on model architecture.

    Args:
        model_arch (str): Type of model architecture like ViT or CMIXER

    Returns:
        tuple: Includes list of files to backup.
    """    
    if model_arch == 'CMIXER':
        file_names_ = ('CMixerDensePrediction.yaml', 'spatiotemporal_cmixer_dense_e1.py')
    elif model_arch == 'ViT':
        file_names_ = ('VitDensePrediction.yaml', 'spatiotemporal_vit_dense_e1.py')
    elif model_arch == 'STAE':
        file_names_ = ('AEDensePrediction.yaml', 'spatiotemporal_ae_dense_e1.py')        
    elif model_arch == 'SwinTransformer':
        file_names_ = ('SwinTransformerV2DensePrediction.yaml', 'spatiotemporal_swinT_dense_e1.py')  
    else:
        file_names_ = ('DiffusionDensePrediction.yaml', 'spatiotemporal_diffusion_dense_e1.py')
    
    return file_names_