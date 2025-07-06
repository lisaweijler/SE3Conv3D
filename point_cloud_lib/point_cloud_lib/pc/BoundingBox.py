import numpy as np
import torch
from torch_scatter import scatter_max, scatter_min


class BoundingBox(object):
    """Class to represent a bounding box for a point cloud.
    """

    def __init__(self, p_point_cloud):
        """Constructor.

        Args:
            p_point_cloud (Pointcloud): Point cloud object.
        """

        self.max_ = scatter_max(p_point_cloud.pts_, p_point_cloud.batch_ids_.to(torch.int64), dim=0)[0] + 1e-6
        self.min_ = scatter_min(p_point_cloud.pts_, p_point_cloud.batch_ids_.to(torch.int64), dim=0)[0] - 1e-6


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Min:\n{}\n"\
                "### Max:\n{}"\
                .format(self.min_, self.max_)