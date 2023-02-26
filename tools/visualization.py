import numpy as np
from typing import Tuple
from nibabel.nifti1 import Nifti1Image
from nilearn import image
from scipy import ndimage


def get_coordinates(inp_img: Nifti1Image, state=False) -> Tuple:
    """Function to find original/MNI coordinate of amplitude

    Args:
        inp_img (Nifti1Image): a 3D Niimg-like object
        state (bool, optional): True if image data contains negative values, otherwise False. Defaults to False.

    Raises:
        TypeError: if 'inp_img' is not a Niimg-like object

    Returns:
        Tuple: a tuple including real and MNI coordinates of amplitude
    """
    if not hasattr(inp_img, 'get_fdata'):
        raise TypeError("Input image is not a Nifti1Image file, please check your input!")
    data = inp_img.get_fdata()
    assert data.ndim == 3, 'Input image must be a 3D tensor (x, y, z)'
    if state:
        data = np.abs(data)
    key_points = ndimage.maximum_position(data)
    return key_points, image.coord_transform(*key_points, inp_img.affine) # type: ignore
