import numpy as np
import time
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class AugPipeline:
    """Class to represent a augmentation pipeline.
    """

    def __init__(self):
        """Constructor.
        """
        self.aug_classes_ = {
            sub.__name__: sub for sub in Augmentation.__subclasses__()}
        self.pipeline_ = []
        
    
    def create_pipeline(self, p_dict_list):
        """Create the pipeline.

        Args:
            p_dict_list (list of dict): List of dictionaries with the
                different augmentations in the pipeline.
        """
        self.pipeline_ = []
        for cur_dict in p_dict_list:
            self.pipeline_.append(self.aug_classes_[cur_dict['name']](**cur_dict))

        
    def increase_epoch_counter(self):
        """Method to update the epoch counter for user-defined augmentations.
        """
        for cur_aug in self.pipeline_:
            cur_aug.increase_epoch_counter() 

    
    def reset_epoch_counter(self):
        """Method to update the epoch counter for user-defined augmentations.
        """
        for cur_aug in self.pipeline_:
            cur_aug.reset_epoch_counter() 
            
        
    def augment(self, p_tensor, p_extra_tensors = []):
        """Method to augment a tensor.

        Args:
            p_tensor (tensor): Input tensor.
            p_extra_tensors (list): List of extra tensors.

        Return:
            aug_tensor (tensor): Augmented tensor.
            params (list of tuples): List of parameters selected for each step of the augmentation.
            aug_extra_tensors (list): List of augmented extra tensors.
        """
        
        cur_tensor = p_tensor
        cur_extra_tensors = p_extra_tensors
        aug_param_list = []
        for cur_aug in self.pipeline_:
            if torch.rand(1).item() <= cur_aug.prob_: 
                cur_tensor, cur_params, cur_extra_tensors = cur_aug.__compute_augmentation__(
                    cur_tensor, cur_extra_tensors)
                aug_param_list.append((cur_aug.__class__.__name__, cur_params))
        return cur_tensor, aug_param_list, cur_extra_tensors