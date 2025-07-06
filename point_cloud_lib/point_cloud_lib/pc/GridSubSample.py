from abc import ABC, abstractmethod
import numpy as np
import torch

from torch_scatter import scatter_max, scatter_mean, scatter_add

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.pc import Grid
from point_cloud_lib.pc import SubSample

class GridSubSample(SubSample):
    """Class to represent a farthest point sampling algorithm.
    """

    def __init__(self, 
        p_pc_src,
        p_cell_size,
        p_rnd_sample = False):
        """Constructor.

        Args:
            p_pc_src (Pointcloud): Source point cloud.
            p_cell_size (float): Cell size.
            p_rnd_sample (bool): Boolean to select a single point from each cell.
        """

        # Store variables.
        self.cell_size_ = p_cell_size
        self.rnd_sample_ = p_rnd_sample

        # Super class init.
        super(GridSubSample, self).__init__(
            p_pc_src)


    def __compute_subsample__(self):
        """Abstract mehod to implement the sub-sample algorithm.
        """
        
        self.grid_ = Grid(self.pc_src_, self.cell_size_)

        if self.rnd_sample_:
            num_pts_cells = scatter_add(
                torch.ones_like(self.grid_.sorted_cell_ids_, dtype=torch.int32), 
                self.grid_.sorted_cell_ids_, dim=0)

            accum_num_pts_cells = torch.cumsum(num_pts_cells, 0)
            accum_num_pts_cells = torch.nn.functional.pad(
                input=accum_num_pts_cells, 
                pad=(1, 0), mode='constant', value=0)[:-1]

            self.ids_ = torch.rand(num_pts_cells.shape[0]).to(num_pts_cells.device)*num_pts_cells
            self.ids_ = torch.floor(self.ids_).to(torch.int32) + accum_num_pts_cells


    def __subsample_tensor__(self, p_tensor, p_method = "avg"):
        """Abstract mehod to implement the sub-sample of a tensor.

        Args:
            p_tensor (tensor): Tensor to sub-sample.
            p_method (string): Sub-sample method.

        Return:
            (tensor): Sub-sampled tensor.
        """
        if self.rnd_sample_:
            return p_tensor[self.grid_.sorted_ids_[self.ids_]]
        else:
            if p_method == "avg":
                return scatter_mean(p_tensor, self.grid_.cell_ids_.to(torch.int64), dim=0)
            elif p_method == "max":
                return scatter_max(p_tensor, self.grid_.cell_ids_.to(torch.int64), dim=0)[0]


    def __upsample_tensor__(self, p_tensor):
        """Abstract method to implement the up-sample of a tensor.

        Args:
            p_tensor (tensor): Tensor to sub-sample.

        Return:
            (tensor): Sub-sampled tensor.
        """
        if self.rnd_sample_:
            scatter_ids = self.grid_.sorted_ids_[self.ids_]
            scatter_ids = scatter_ids.unsqueeze(1).repeat(1, p_tensor.shape[-1])
            upsampled_tensor = torch.zeros(
                (self.grid_.sorted_cell_ids_.shape[0], p_tensor.shape[-1]),
                dtype = p_tensor.dtype).to(p_tensor.device).scatter_(
                    0, scatter_ids, p_tensor)
            return upsampled_tensor
        else:
            return p_tensor[self.grid_.cell_ids_]

