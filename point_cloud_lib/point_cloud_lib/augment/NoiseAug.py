from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class NoiseAug(Augmentation):
    """Class to represent a tensor augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_stddev = 0.005,
                 p_clip = None, 
                 p_apply_extra_tensors=[],
                 **kwargs):
        """Constructor.

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_stddev (float): Stddev of the noise.
            p_clip (float): Maximum value to clip.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.stddev_ = p_stddev
        self.clip_ = p_clip

        # Super class init.
        super(NoiseAug, self).__init__(p_prob, p_apply_extra_tensors)


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
        cur_noise = torch.randn(
            p_tensor.shape, dtype=p_tensor.dtype).to(device)*self.stddev_
        if not self.clip_ is None:
            cur_noise = torch.clip(cur_noise, min=-self.clip_, max=self.clip_)
        aug_tensor = p_tensor + cur_noise 

        # Extra tensors.
        new_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                new_extra_tensors.append(
                    cur_tensor + cur_noise*self.stddev_)
            else:
                new_extra_tensors.append(cur_tensor)

        return aug_tensor, (cur_noise,), new_extra_tensors
