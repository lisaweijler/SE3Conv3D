from abc import ABC, abstractmethod
import numpy as np
import torch

from torch_scatter import scatter_add
from torch_cluster import radius

from point_cloud_lib.pc import Neighborhood
from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.custom_ops import BallQuery

class BQNeighborhood(Neighborhood):
    """Class to represent a ball query neighborhood.
    """

    def __init__(self, 
        p_pc_src, 
        p_samples,
        p_radius,
        p_max_neighbors = 0):
        """Constructor.

        Args:
            p_pc_src (Pointcloud): Source point cloud.
            p_pc_samples (Pointcloud): Sample point cloud.
            p_radius (float): Radius.
            p_max_neighbors (int): Maximum number of neighbors.
        """

        # Store variables.
        self.radius_ = p_radius
        self.max_neighbors_ = p_max_neighbors

        # Super class init.
        super(BQNeighborhood, self).__init__(
            p_pc_src, p_samples)


    def __compute_neighborhood__(self):
        """Abstract mehod to implement the neighborhood selection.
        """
        
        #if self.max_neighbors_ > 0 and self.samples_.pts_.shape[0] < 100000:
        #    self.neighbors_ = radius(
        #        x = self.pc_src_.pts_, y =self.samples_.pts_, 
        #        r=self.radius_, 
        #        batch_x=self.pc_src_.batch_ids_, 
        #        batch_y=self.samples_.batch_ids_, 
        #        max_num_neighbors=self.max_neighbors_)
        #    self.neighbors_ = torch.transpose(self.neighbors_, 0, 1).contiguous()
        #    
        #    self.start_ids_ = scatter_add(
        #        torch.ones_like(self.neighbors_[:,1], dtype=torch.int32), 
        #        self.neighbors_[:,0], dim=0)
        #    self.start_ids_ = torch.cumsum(self.start_ids_, 0)
        #
        #else:
        self.neighbors_, self.start_ids_ = BallQuery.apply(
            self.pc_src_.pts_,
            self.samples_.pts_,
            self.pc_src_.batch_ids_,
            self.samples_.batch_ids_,
            self.radius_,
            self.max_neighbors_)

        #new_start_ids = scatter_add(
        #    torch.ones_like(self.neighbors_[:,1], dtype=torch.int32), 
        #    self.neighbors_[:,0].to(torch.int64), dim=0)
        #old_start_ids = scatter_add(
        #    torch.ones_like(old_neighbors[:,1], dtype=torch.int32), 
        #    old_neighbors[:,0].to(torch.int64), dim=0)
        #print(torch.amax(new_start_ids, 0).item(),
        #        torch.amax(old_start_ids, 0).item())
        #print(self.neighbors_.shape, neighbors.shape,
        #      self.start_ids_.shape, start_ids.shape)
        
