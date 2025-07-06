from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.augment import Augmentation

class RotationAug(Augmentation):
    """Class to represent a point cloud augmentation.
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_axis = 0,
                 p_min_angle= 0,
                 p_max_angle = 2*np.pi,
                 p_angle_values = None,
                 p_apply_extra_tensors=[], 
                 **kwargs):
        """Constructor. 

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_axis (Int): Index of the axis.
            p_min_angle (float): Minimum rotation angle.
            p_max_angle (float): Maximum rotation angle.
            p_angle_values (list of floats): User-defined angle per epoch.
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.
        """

        # Store variables.
        self.axis_ = p_axis
        self.min_angle_ = p_min_angle
        self.max_angle_ = p_max_angle
        self.angle_values_ = p_angle_values

        # Super class init.
        super(RotationAug, self).__init__(p_prob, p_apply_extra_tensors)


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
        if self.angle_values_ is None:
            cur_angle = torch.rand(1).item()*\
                (self.max_angle_-self.min_angle_) + self.min_angle_
        else:
            cur_angle = self.angle_values_[self.epoch_iter_]
        device = p_pts.device
        if self.axis_ == 0:
            R = torch.from_numpy(
                np.array([[1.0, 0.0, 0.0],
                          [0.0, np.cos(cur_angle), -np.sin(cur_angle)],
                          [0.0, np.sin(cur_angle), np.cos(cur_angle)]])).to(device).to(torch.float32)
        elif self.axis_ == 1:
            R = torch.from_numpy(
                np.array([[np.cos(cur_angle), 0.0, np.sin(cur_angle)],
                          [0.0, 1.0, 0.0],
                          [-np.sin(cur_angle), 0.0, np.cos(cur_angle)]])).to(device).to(torch.float32)
        elif self.axis_ == 2:
            R = torch.from_numpy(
                np.array([[np.cos(cur_angle), -np.sin(cur_angle), 0.0],
                          [np.sin(cur_angle), np.cos(cur_angle), 0.0],
                          [0.0, 0.0, 1.0]])).to(device).to(torch.float32)

        aug_pts = torch.matmul(p_pts, R)

        # Extra tensors.
        new_extra_tensors = []
        for cur_iter, cur_tensor in enumerate(p_extra_tensors):
            if self.apply_extra_tensors_[cur_iter]:
                new_extra_tensors.append(
                    torch.matmul(cur_tensor, R))
            else:
                new_extra_tensors.append(cur_tensor)

        return aug_pts, (self.axis_, cur_angle), new_extra_tensors
