import torch
import numpy as np
#from pytorch3d import transforms

from point_cloud_lib.augment import Augmentation
from point_cloud_lib.pc import random_rotation

"""From pytorch3D documentation:

The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""

class RotationAug3D(Augmentation):
    """Class to represent a point cloud augmentation.
       Uniform random 3D rotations. 
    """

    def __init__(self, 
                 p_prob = 1.0,
                 p_apply_extra_tensors=[],
                 p_axis = None, 
                 **kwargs):
        """Constructor. 

        Args:
            p_prob (float): Probability of executing this augmentation.
            p_axis (int): Index of axis to rotat around. If None, rotate random 3D
            p_apply_extra_tensors (list bool): List of boolean indicating
                if the augmentation should be used to the extra tensors.

        Rotations are counterclockwise in a right-handed coordinate system. 
        """
        self.axis_ = p_axis


        # Super class init.
        super(RotationAug3D, self).__init__(p_prob, p_apply_extra_tensors)


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
       
        if self.axis_ is None:
            R = random_rotation(device=p_pts.device)
            

        else:
            device = p_pts.device
            cur_angle = torch.rand(1).item()*2*np.pi
                #(self.max_angle_-self.min_angle_) + self.min_angle_

            # counter clockwise rotations for pre-multiplication of col vectors
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

        return aug_pts, (R,), new_extra_tensors
