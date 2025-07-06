from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class MirrorAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_mirror_prob = 0.5,
                 p_axes = [True, True, False], 
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
        self.axes_ = p_axes
        self.mirror_prob_ = p_mirror_prob

        # Super class init.
        super(MirrorAug, self).__init__(p_prob, p_apply_extra_tensors)


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
        mask_1 = torch.rand(p_pts.shape[-1]).to(device) > self.mirror_prob_
        mask_2 = torch.from_numpy(np.array(self.axes_)).to(device)
        mask = torch.logical_and(mask_1, mask_2)
        mirror_vec = torch.ones(3).to(device)*(1.-mask.to(torch.float32)) - torch.ones(3).to(device)*mask 

        aug_pts = p_pts*mirror_vec.reshape((1, -1))

        # Extra tensors.
        new_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                new_extra_tensors.append(
                    cur_tensor*mirror_vec.reshape((1, -1)))
            else:
                new_extra_tensors.append(cur_tensor)

        return aug_pts, (mirror_vec,), new_extra_tensors
