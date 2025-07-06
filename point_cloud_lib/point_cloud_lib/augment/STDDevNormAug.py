from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class STDDevNormAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self,
                 p_new_std = 1.0,
                 p_apply_extra_tensors=[],
                 **kwargs):
        """Constructor.

        Args:
            p_max_aabb_ratio (float): Maximum ratio of displacement
                wrt. the bounding box.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.stddev_ = p_new_std

        # Super class init.
        super(STDDevNormAug, self).__init__(1.0, p_apply_extra_tensors)


    def __compute_augmentation__(self,
                                 p_pts,
                                 p_extra_tensors = []):
        """Abstract method to implement the augmentation.

        Args:
            p_tensor (tensor): Input tensor.
            
        Return:
            aug_tensor (tensor): Augmented tensor.
            params (tuple): Parameters selected for the augmentation.
            p_extra_tensors (list): List of extra tensors.
        """
        prev_stddev = torch.amax(torch.std(p_pts, 0))
        aug_pts = (p_pts*self.stddev_)/prev_stddev

        # Extra tensors.
        new_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                new_extra_tensors.append(
                    (cur_tensor*self.stddev_)/prev_stddev)
            else:
                new_extra_tensors.append(cur_tensor)
        
        return aug_pts, (prev_stddev, self.stddev_), new_extra_tensors
