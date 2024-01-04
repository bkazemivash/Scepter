"""
This module includes evaluation methods for different scenarios like 
dense prediciton or recognition.
"""

import torch
from torch.nn import CosineSimilarity
from multipledispatch import dispatch



def iou_3D(outputs: torch.Tensor, labels: torch.Tensor, thr = (.5, .5), smooth_ratio = 1e-6) -> float:
    """Computes Intersection Over Union (IOU) metric based on number of 
        voxels in intersection and union of ROI. Masking threshold is used
        to seperated forground from background.

    Args:
        outputs (torch.Tensor): 3D tensor including predicted score map (x,y,z).
        labels (torch.Tensor): 3D tensor including prior map  (x,y,z).
        thr (tuple, optional): Masking threshold for output and label. Defaults to (.5, .5).
        smooth_ratio (_type_, optional): Epsilon value to prevent devided by zero case. Defaults to 1e-6.

    Returns:
        float: Voxel-wise IOU of 2 maps.
    """
    assert (outputs.ndim == 3) and (labels.ndim ==3), '3D tensors are expected while inputs have different dimension.'
    outputs = (outputs>thr[0]).int()
    labels = (labels>thr[1]).int()
    intersection = (outputs & labels).float().sum()  
    union = (outputs | labels).float().sum()            
    iou = (intersection + smooth_ratio) / (union + smooth_ratio)      
    
    return iou  


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