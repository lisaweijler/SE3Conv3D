import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.pc import BoundingBox
from point_cloud_lib.custom_ops import ComputeKeys

class Grid(object):
    """Class to distribute points into a regular grid.
    """

    def __init__(self, p_point_cloud, p_cell_size):
        """Constructor.

        Args:
            p_point_cloud (Pointcloud): Input point cloud.
            p_cell_size (float): Cell size.
        """
        
        # Save the parameters.
        self.pointcloud_ = p_point_cloud
        self.bounding_box_ = BoundingBox(self.pointcloud_)
        self.cell_size_ = p_cell_size

        # Compute the number of cells.
        num_cells = (self.bounding_box_.max_ - self.bounding_box_.min_)/self.cell_size_
        self.num_cells_ = torch.max(num_cells.to(torch.int32) + 1, dim=0)[0]
        
        # Initialize the grid tensors.
        self.cell_ids_ = None
        self.sorted_ids_ = None
        self.sorted_cell_ids_ = None

        # Compute cell ids.
        self.__compute_cell_ids__()


    def __compute_cell_ids__(self):
        """ Method to compute the cell ids.
        """

        # Compute cell ids.
        self.cell_ids_ = ComputeKeys.apply(
            self.pointcloud_.pts_,
            self.pointcloud_.batch_ids_,
            self.bounding_box_.min_,
            self.num_cells_,
            torch.from_numpy(np.array(
                [self.cell_size_ for _ in range(self.num_cells_.shape[0])]))\
                    .to(torch.float32).to(self.num_cells_.device)
        )
        _, self.cell_ids_ = torch.unique(self.cell_ids_, return_inverse=True)
        
        # Sort the indices.
        self.sorted_ids_ = torch.argsort(self.cell_ids_)

        # Sorted cell ids.
        self.sorted_cell_ids_ = self.cell_ids_[self.sorted_ids_ ]


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Cell size:\n{}\n"\
                "### Num cells:\n{}\n"\
                "### Cell Ids:\n{}\n"\
                "### Sorted Ids:\n{}\n"\
                .format(self.cell_size_, 
                self.num_cells_, self.cell_ids_,
                self.sorted_ids_)