from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class DropAug(Augmentation):
    """Class to represent a tensor augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_apply_extra_tensors=[],
                 p_drop_prob = 0.05,
                 p_keep_zeros = True, 
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
            p_prob (float): Remove probability.
            p_keep_zeros (bool): Boolean that indicates if the 
                drop values are kept as zeros.
        """

        # Store variables.
        self.drop_prob_ = p_drop_prob
        self.keep_zeros_ = p_keep_zeros

        # Super class init.
        super(DropAug, self).__init__(p_prob, p_apply_extra_tensors)


    def __compute_augmentation__(self,
                                 p_tensor,
                                 p_extra_tensors = []):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_extra_tensors (list): List of extra tensors.
        """
        device = p_tensor.device
        mask = torch.rand(p_tensor[:, 0].shape).to(p_tensor.device) > self.drop_prob_
        
        if self.keep_zeros_:
            aug_tensor = p_tensor*mask + torch.ones_like(p_tensor)*(1.-mask)
            aug_extra_tensors = []
            for cur_iter, cur_tensor in enumerate(p_extra_tensors):
                if self.apply_extra_tensors_[cur_iter]:
                    aug_extra_tensors.append(cur_tensor*mask + torch.ones_like(cur_tensor)*(1.-mask))
                else:
                    aug_extra_tensors.append(cur_tensor)
        else:
            aug_tensor = p_tensor[mask]
            aug_extra_tensors = []
            for cur_iter, cur_tensor in enumerate(p_extra_tensors):
                if self.apply_extra_tensors_[cur_iter]:
                    aug_extra_tensors.append(cur_tensor[mask])
                else:
                    aug_extra_tensors.append(cur_tensor)

        return aug_tensor, (mask,), aug_extra_tensors
