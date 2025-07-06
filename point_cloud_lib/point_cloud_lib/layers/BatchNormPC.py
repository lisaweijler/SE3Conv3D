from abc import ABC, abstractmethod
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.layers import NormLayerPC

class BatchNormPC(NormLayerPC):
    """Class to represent a normalization layer.
    """

    def __init__(self, p_num_features):
        """Constructor.

        Args:
            p_num_features (int): Number of features.
        """

        # Super class init.
        super(BatchNormPC, self).__init__(p_num_features)
        
        # Create the norm layer.
        self.layer_ = torch.nn.BatchNorm1d(p_num_features, momentum=0.2)


    def forward(self, p_x, p_pc):
        """Forward method.

        Args:
            p_x (tensor): Input tensor.
            p_pc (Pointcloud): Input point cloud.
        """
        return self.layer_(p_x)