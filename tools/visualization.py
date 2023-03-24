import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch, os
from datetime import datetime
from typing import Tuple, Dict, Union, Any
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

def draw_confusion_matrix(inp_matrix: torch.Tensor, 
                          cls_names: Dict,
                          show_percentage: bool = False,
                          save_dir: Union[None, str] = None) -> Any:
    """Drawing confusion matrix.

    Args:
        inp_matrix (torch.Tensor): The given confusion matrix.
        cls_names (Dict): Class names.
        show_percentage (bool): Shows percentage of each class if True. Defaults to False.
        save_dir (Union[None, str], optional): If not None, save the figure. Defaults to None.

    Returns:
        Any: Returns a matplotlib figure.
    """    
    assert inp_matrix.shape == tuple((len(cls_names), len(cls_names))) , 'Invalid class names for the given confusion matrix'
    data_ = inp_matrix.cpu().detach().numpy()
    if show_percentage:
        data_ /= np.sum(data_, axis=0)
    labels = list(cls_names.keys())
    plt.figure(figsize=(6,6), dpi=100)
    sns.set(font_scale = .8)
    ax = sns.heatmap(data_, annot=True, fmt='.3%', cmap='Blues')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.set_xlabel("Actual Diagnosis", fontsize=14, labelpad=20)
    ax.set_ylabel("Predicted Diagnosis", fontsize=14, labelpad=20)
    ax.set_title ("Confusion Matrix for the Brain Disease Detection Model", fontsize=14, pad=20)
    fig = ax.get_figure()
    if save_dir:
        address_ = os.path.join(save_dir,'figures', f'ConfusionMatrix_{datetime.now().strftime("%y%m%d_%H%M%S")}.png')
        fig.savefig(address_, bbox_inches="tight")
    return fig