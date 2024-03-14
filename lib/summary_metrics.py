"""
This module includes evaluation methods for different scenarios like 
dense prediciton or recognition.
"""

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from math import exp



def iou_3D(outputs: torch.Tensor, prior: torch.Tensor, thr = (.5, .5), smooth_ratio = 1e-6) -> float:
    """Computes Intersection Over Union (IOU) metric based on number of 
        voxels in intersection and union of ROI. Masking threshold is used
        to seperated forground from background.

    Args:
        outputs (torch.Tensor): 3D tensor including predicted score map (x,y,z).
        prior (torch.Tensor): 3D tensor including prior map  (x,y,z).
        thr (tuple, optional): Masking threshold for output and label. Defaults to (.5, .5).
        smooth_ratio (_type_, optional): Epsilon value to prevent devided by zero case. Defaults to 1e-6.

    Returns:
        float: Voxel-wise IOU of 2 maps.
    """
    assert (outputs.ndim == 3) and (prior.ndim ==3), '3D tensors are expected while inputs have different dimensions.'
    outputs = (outputs>thr[0]).int()
    prior = (prior>thr[1]).int()
    intersection = (outputs & prior).float().sum()  
    union = (outputs | prior).float().sum()            
    iou = (intersection + smooth_ratio) / (union + smooth_ratio)      
    
    return iou  


def mare_3D(output: torch.Tensor, prior: torch.Tensor, thr=0.) -> float:
    """Computes mean absolute relative error for the given tensors (score maps)

    Args:
        output (torch.Tensor): 3D tensor including predicted score map (x,y,z)
        prior (torch.Tensor): 3D tensor including prior map (x,y,z)
        thr (float, optional): Masking threshold for output and prior. Defaults to 0.

    Returns:
        float: averaged absolute relative error over given maps.
    """
    assert (output.ndim != 3) and (prior.ndim != 3), '3D tensors are expected while inputs have different dimensions.'
    mask = torch.gt(prior, thr)
    error_ = torch.abs(output[mask] - prior[mask]) / torch.abs(prior[mask])    
    return error_.mean()


def ssim_3D(img1: torch.Tensor, img2: torch.Tensor, window_size: int, sigma=1.5, channel=1, reduction: str = 'mean') -> float:
    """Computes structural similarity index measure (SSIM) for the given images.

    Args:
        img1 (torch.Tensor): 3D tensor including predicted score map (x,y,z)
        img2 (torch.Tensor): 3D tensor including prior map (x,y,z)
        window_size (int): Size of window for convolution layers.
        sigma (float, optional): Sigma coefficient od normal distribution. Defaults to 1.5.
        channel (int, optional): Input channel size of images. Defaults to 1.
        reduction (str, optional): Reduction policy on output. Defaults to mean.

    Returns:
        float: SSIM metric for the input images.
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)
        ]
    )
    if img1.is_cuda:
         gauss = gauss.cuda(img1.get_device())
    gauss = gauss / gauss.sum()

    window_1D = gauss.unsqueeze(1)
    window_2D = window_1D.mm(window_1D.t())
    window_3D = (
        window_1D.mm(window_2D.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window_m = window_3D.expand(channel, 1, window_size, window_size, window_size).contiguous().type_as(img1)
    
    n_window_size = int(window_size // 2)
    mu1 = F.conv3d(img1, window_m, padding=n_window_size, groups=channel)
    mu2 = F.conv3d(img2, window_m, padding=n_window_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv3d(img1 * img1, window_m, padding=n_window_size, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv3d(img2 * img2, window_m, padding=n_window_size, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv3d(img1 * img2, window_m, padding=n_window_size, groups=channel) - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_ = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if reduction == 'mean':
         return ssim_.mean()
    elif reduction == 'space_mean':
        return ssim_.mean(dim=(1,2,3,4))
    elif reduction == 'None':
        return ssim_    
    else:
         raise ValueError('Choose a reduction scenario from [mean, space_mean, None].')


def robust_scaling(x1: torch.Tensor):
    """Implentation of Robust Scaling method - a median-based scaling method.

    Args:
        x1 (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Scaled tensor.
    """
    Q1 = torch.quantile(x1, .25, dim=0,)
    Q3 = torch.quantile(x1, .75, dim=0,)
    M,_ = torch.median(x1, dim=0,)
    IQR = Q3 - Q1
    scaled_data = (x1 - M.squeeze(0)) / IQR.squeeze(0)
    
    return scaled_data


def compute_accuracy(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the top 1 accuracy on given tensors.

    Args:
        x1 (torch.Tensor): Output of the model.
        x2 (torch.Tensor): Target value.

    Returns:
        torch.Tensor: Accuracy metric for the given tensors.

    """   
    _,target = x1.topk(1, dim=1)
    return torch.sum(target == x2)


def compute_correlation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the correlation between given tensors.
    
    Args:
        x1 (torch.Tensor): Output of the model.
        x2 (torch.Tensor): Target value.

    Returns:
        torch.Tensor: Correlation between the given tensors.

    """   
    cos = CosineSimilarity(dim=1, eps=1e-6)
    pearson = cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))
    return pearson


def compute_confusion_matrix(x1: torch.Tensor, 
                             x2: torch.Tensor, 
                             n_cls: int) -> torch.Tensor:
    """Computing confusion matrix.

    Args:
        x1 (torch.Tensor): Predicted labels.
        x2 (torch.Tensor): Ground truth labels.
        n_cls (int): Number of classes.

    Returns:
        torch.Tensor: Confusion matrix for the given tensors.
    """    
    confusion_matrix = torch.zeros(n_cls, n_cls)
    if n_cls == 2:
         target = x1.sigmoid().round()
    else:
         _,target = x1.topk(1, dim=1)  

    for t, p in zip(target.view(-1), x2.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1      
    return confusion_matrix