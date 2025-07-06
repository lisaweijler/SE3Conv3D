from abc import ABC, abstractmethod
import numpy as np
import scipy.interpolate
import scipy.ndimage

import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class ElasticDistortionAug(Augmentation):
    """Class to represent a elastic distortion point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_granularity = [0.1],
                 p_magnitude = [0.2],
                 p_apply_extra_tensors=[],
                 **kwargs):
        """Constructor. Augmentation y = x*a + b

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_granularity (float): Granularity.
            p_magnitude (float): Magnitude.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.granularity_ = p_granularity
        self.magnitude_ = p_magnitude

        # Super class init.
        super(ElasticDistortionAug, self).__init__(p_prob, p_apply_extra_tensors)


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
        cur_type = p_tensor.dtype
        
        coords = p_tensor.detach().clone()

        blurx = torch.ones((3, 1, 3, 1, 1)).to(cur_type).to(device) / 3
        blury = torch.ones((3, 1, 1, 3, 1)).to(cur_type).to(device) / 3
        blurz = torch.ones((3, 1, 1, 1, 3)).to(cur_type).to(device) / 3

        coords_min = torch.amin(coords, 0).reshape((1, -1))
        coords_max = torch.amax(coords, 0).reshape((1, -1))
        noise_dims_full = torch.amax(coords - coords_min, 0)
        
        for cur_granularity, cur_magnitude in zip(self.granularity_, self.magnitude_):
            
            noise_dim = (noise_dims_full // cur_granularity).to(torch.int32) + 3
            noise = torch.randn(1, 3, *noise_dim).to(cur_type).to(device)

            # Smoothing.
            convolve = torch.nn.functional.conv3d
            for _ in range(2):
                noise = convolve(noise, blurx, padding='same', groups=3)
                noise = convolve(noise, blury, padding='same', groups=3)
                noise = convolve(noise, blurz, padding='same', groups=3)

            # Trilinear interpolate noise filters for each spatial dimensions.
            sample_coords = ((coords - coords_min)/(coords_max - coords_min))*2. - 1.
            sample_coords = sample_coords.reshape(1, -1, 1, 1, 3) # [N, 1, 1, 3]
            new_sample_coords = sample_coords.clone()
            new_sample_coords[..., 0] = sample_coords[..., 2]
            new_sample_coords[..., 2] = sample_coords[..., 0]
            sample = torch.nn.functional.grid_sample(
                noise, new_sample_coords, align_corners=True, 
                padding_mode='border')[0,:,:,0,0].transpose(0,1)

            coords += sample * cur_magnitude

        aug_tensor = coords

        return aug_tensor, None, p_extra_tensors
