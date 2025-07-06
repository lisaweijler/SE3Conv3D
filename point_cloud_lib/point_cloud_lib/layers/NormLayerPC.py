from abc import ABC, abstractmethod
import torch

from point_cloud_lib.pc import Pointcloud

class NormLayerPC(torch.nn.Module, ABC):
    """Class to represent a normalization layer.
    """

    def __init__(self, p_num_features):
        """Constructor.

        Args:
            p_num_features (int): Number of features.
        """

        # Super class init.
        super(NormLayerPC, self).__init__()

        # Store parameters.
        self.num_feats_ = p_num_features


    @abstractmethod
    def forward(self, p_x, p_pc):
        """Forward method.

        Args:
            p_x (tensor): Input tensor.
            p_pc (Pointcloud): Input point cloud.
        """
        pass