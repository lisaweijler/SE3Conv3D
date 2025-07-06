from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class TranslationAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_max_aabb_ratio = 1.0, 
                 p_apply_extra_tensors=[],
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_max_aabb_ratio (float): Maximum ratio of displacement
                wrt. the bounding box.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.max_aabb_ratio_ = p_max_aabb_ratio

        # Super class init.
        super(TranslationAug, self).__init__(p_prob, p_apply_extra_tensors)


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
        device = p_pts.device
        cur_translation = (torch.rand(p_pts.shape[-1]).to(device)*2. - 1.)*self.max_aabb_ratio_
        min_pt = torch.amin(p_pts, 0)
        max_pt = torch.amax(p_pts, 0)
        displacement_vec = (max_pt - min_pt)/2. * cur_translation
        aug_pts = p_pts + displacement_vec.reshape((1, -1))

        # Extra tensors.
        new_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                new_extra_tensors.append(
                    cur_tensor + displacement_vec.reshape((1, -1)))
            else:
                new_extra_tensors.append(cur_tensor)

        return aug_pts, (displacement_vec,), new_extra_tensors
