from abc import ABC, abstractmethod
import numpy as np
import torch

from torch_cluster import fps

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.pc import SubSample

class FPSSubSample(SubSample):
    """Class to represent a farthest point sampling algorithm.
    """

    def __init__(self, 
        p_pc_src,
        p_ratio):
        """Constructor.

        Args:
            p_pc_src (Pointcloud): Source point cloud.
            p_ratios (float): Ratio of points selected
        """

        # Store variables.
        self.ratio_ = p_ratio

        # Super class init.
        super(FPSSubSample, self).__init__(
            p_pc_src)


    def __compute_subsample__(self):
        """Abstract mehod to implement the sub-sample algorithm.
        """
        
        self.ids_ = fps(
            self.pc_src_.pts_, self.pc_src_.batch_ids_, self.ratio_)
        
    
    def __subsample_tensor__(self, p_tensor, p_method = "avg"):
        """Abstract mehod to implement the sub-sample of a tensor.

        Args:
            p_tensor (tensor): Tensor to sub-sample.
            p_method (string): Sub-sample method.

        Return:
            (tensor): Sub-sampled tensor.
        """
        return p_tensor[self.ids_]


    def __upsample_tensor__(self, p_tensor):
        """Abstract method to implement the up-sample of a tensor.

        Args:
            p_tensor (tensor): Tensor to sub-sample.

        Return:
            (tensor): Sub-sampled tensor.
        """
        pass #TODO