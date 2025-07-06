from abc import ABC, abstractmethod
import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud

class Neighborhood(ABC):
    """Class to represent a neighborhood.
    """

    def __init__(self, 
        p_pc_src, 
        p_samples):
        """Constructor.

        Args:
            p_pc_src (Pointcloud): Source point cloud.
            p_pc_samples (Pointcloud): Sample point cloud.
        """

        # Store variables.
        self.pc_src_ = p_pc_src
        self.samples_ = p_samples

        # Init the neighborhood tensors.
        self.neighbors_ = None 
        self.start_ids_ = None

        # Compute neighborhood.
        self.__compute_neighborhood__()


    @abstractmethod
    def __compute_neighborhood__(self):
        """Abstract mehod to implement the neighborhood selection.
        """
        pass


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Neighbors:\n{}\n"\
                "### Start indices:\n{}"\
                .format(self.neighbors_, self.start_ids_)
