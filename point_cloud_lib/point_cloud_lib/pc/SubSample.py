from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud

class SubSample(ABC):
    """Class to represent a sub-sample algorithm.
    """

    def __init__(self, 
        p_pc_src):
        """Constructor.

        Args:
            p_pc_src (Pointcloud): Source point cloud.
        """

        # Store variables.
        self.pc_src_ = p_pc_src

        # Init the id tensors.
        self.ids_ = None

        # Compute neighborhood.
        self.__compute_subsample__()


    @abstractmethod
    def __compute_subsample__(self):
        """Abstract mehod to implement the sub-sample algorithm.
        """
        pass


    @abstractmethod
    def __subsample_tensor__(self, p_tensor, p_method = "avg"):
        """Abstract mehod to implement the sub-sample of a tensor.

        Args:
            p_tensor (tensor): Tensor to sub-sample.
            p_method (string): Sub-sample method.

        Return:
            (tensor): Sub-sampled tensor.
        """
        pass


    @abstractmethod
    def __upsample_tensor__(self, p_tensor):
        """Abstract method to implement the up-sample of a tensor.

        Args:
            p_tensor (tensor): Tensor to sub-sample.

        Return:
            (tensor): Sub-sampled tensor.
        """
        pass


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Ids:\n{}\n"\
                .format(self.ids_)
