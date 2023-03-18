"""
Helper functions for verification and preprocessing.
This module contains mandatory functions to run different preprocessing and 
postprocessing steps on input fMRI images.
"""

import numpy as np
import operator
import torch
import torch.nn as nn
from typing import Tuple, Union, Dict, Any
from nilearn.masking import unmask, apply_mask
from nilearn.image import index_img
from nibabel.nifti1 import Nifti1Image
from scipy import stats
from functools import reduce


def weights_init(m: nn.Module) -> None:
    """ Initialize model weights using xavier method

    Args:
        m (nn.Module): model layers
    """
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
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


def get_num_patches(inp_dim: Tuple[int, ...], patch_size: int) -> int:
    """ Computes number of patches for a given tensor.
       
    Args:
        inp_dim (Tuple[int, ...]): input tuple
        patch_size (int): size of the patch

    Returns:
        int: Number of generated patches
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


def normalize_array(ar: np.ndarray, ax = -1) -> np.ndarray:
    """ Z-score input array

    Args:
        ar (np.ndarray): input array to be normalized.         
        ax (obj:int, optional): axis to apply scaling function. Defaults to None.

    Returns:
        np.ndarray: normalized (z-score) array
    """
    return stats.zscore(ar, axis=ax)


def fmri_preprocess(inp_img: Union[str, Nifti1Image],
                    mask_img: str,
                    norm_dim: Union[None, int] = None,
                    scale: Union[None, Tuple[int, int]] = None,
                    time_slice =0,
                    step_size=1) -> np.ndarray:
    """ Mask, z-score, and scale input fMRI image.

    Args:
        inp_img (Union[str, Nifti1Image]): A 4D Niimg-like object or a directory of it.
        mask_img (str): Path to a 3D Niimg-like mask object
        norm_dim (Union[None, int], optional): Z-score by a specific axis; 0 for voxel-wise(fMRI), 1 for timepoint-wise(fMRI). Defaults to None.
        scale (Union[None, Tuple[int, int]], optional): True if scaling is needed range: [lower, upper]. Defaults to None.
        time_slice (int, optional): Slice of timepoints. Defaults to 0.
        step_size (int, optional): Sampling rate of timepoints. Defaults to 1.

    Raises:
        TypeError: If 'inp_img' is not a Niimg-like object

    Returns:
        np.ndarray: 4D image array
    """    
    if ((not isinstance(inp_img, Nifti1Image)) and (isinstance(inp_img, str) and not (inp_img.lower().endswith(('.nii', '.nii.gz'))))):
        raise TypeError("Input image is not a Nifti file, please check your input!") 
    totall_timepoints = time_slice * step_size
    if time_slice > 0:
        inp_img = index_img(inp_img, slice(0, totall_timepoints, step_size)) # type: ignore    
    data = apply_mask(inp_img, mask_img)
    if norm_dim is not None:
        data = normalize_array(data, ax=norm_dim)
    if scale:
        data = scale_array(data, lb=scale[0], ub=scale[1], ax=-1)
    data = unmask(data, mask_img)
    return data.get_fdata() # type: ignore


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

def to_index(class_labels: np.ndarray) -> Dict:
    """Make a dict with keys:labels to value: index

    Args:
        class_labels (np.ndarray): list of class labels

    Returns:
        Dict: key, value : label, index
    """    
    label_index_dict = {label: index for index, label in enumerate(class_labels)}
    return label_index_dict

def compute_class_weights(x: Any) -> torch.Tensor:
    """Compute class weights in an imbalanced dataset

    Args:
        x (Any): Array-like object including classes 

    Returns:
        torch.Tensor: Class weight tensor
    """    
    class_count = np.unique(x, return_counts=True)[1]
    weights = 1. / class_count
    weights = np.asarray([.1, .45, .45])
    return torch.tensor(weights, dtype=torch.float)