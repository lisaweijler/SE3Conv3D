from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud

class Augmentation(ABC):
    """Class to represent a tensor augmentation.
    """

    def __init__(self, p_prob, p_apply_extra_tensors, **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """
        self.prob_ = p_prob
        self.apply_extra_tensors_ = p_apply_extra_tensors
        self.epoch_iter_ = 0


    def increase_epoch_counter(self):
        """Method to update the epoch counter for user-defined augmentations.
        """
        self.epoch_iter_ += 1 


    def reset_epoch_counter(self):
        """Method to update the epoch counter for user-defined augmentations.
        """
        self.epoch_iter_ = 0    
    
    
    @abstractmethod
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
        pass
