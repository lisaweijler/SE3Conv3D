from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class CropPtsAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_max_pts = 0,
                 p_crop_ratio = 1.0, 
                 p_apply_extra_tensors=[],
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_max_pts (float): Maximum number of points.
            p_crop_ratio (float): Scene crop ratio.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.max_pts_ = p_max_pts
        self.crop_ratio_ = p_crop_ratio

        # Super class init.
        super(CropPtsAug, self).__init__(p_prob, p_apply_extra_tensors)


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
        cur_num_pts = p_pts.shape[0]
        max_num_pts = self.max_pts_ if self.max_pts_ > 0 else cur_num_pts
        max_num_pts = min(max_num_pts, int(cur_num_pts * self.crop_ratio_))

        crop_mask = torch.ones(cur_num_pts, dtype=torch.bool).to(p_pts.device)
        if cur_num_pts > max_num_pts:
            rand_idx = torch.randint(low=0, high=cur_num_pts, size=(1,)).to(p_pts.device)
            sort_idx = torch.argsort(torch.sum((p_pts - p_pts[rand_idx])**2, 1))
            crop_idx = sort_idx[max_num_pts:]
            crop_mask[crop_idx] = False
            aug_pts = p_pts[crop_mask]
        else:
            aug_pts = p_pts


        # Extra tensors.
        aug_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                aug_extra_tensors.append(cur_tensor[crop_mask])
            else:
                aug_extra_tensors.append(cur_tensor)

        return aug_pts, (crop_mask), aug_extra_tensors
