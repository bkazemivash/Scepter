import torch
import torch.nn as nn
from torch.nn import CosineSimilarity

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

def compute_roc_curve(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.rand(1)

def robust_scaling(x1: torch.Tensor):
    # Calculate the first quartile (Q1) and the third quartile (Q3)
    Q1 = torch.quantile(x1, .25, dim=0,)
    Q3 = torch.quantile(x1, .75, dim=0,)
    M,_ = torch.median(x1, dim=0,)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    
    # Apply Robust Scaling
    scaled_data = (x1 - M.squeeze(0)) / IQR.squeeze(0)
    
    return scaled_data