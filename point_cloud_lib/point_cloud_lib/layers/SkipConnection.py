import torch

from point_cloud_lib.pc import Pointcloud

from .DropPathPC import DropPathPC

class SkipConnection(torch.nn.Module):
    """Class to implement a skip connection for point clouds.
    """

    def __init__(self,  
        p_drop_prob,
        p_num_features,
        p_init_gamma= 1e-6):
        """Constructor.

        Args:
            p_drop_prob (float): Drop probability
            p_num_features (int): Number of features.
            p_init_gamma (float): Initial value of gamma.
        """

        # Super constructor.
        torch.nn.Module.__init__(self)
        
        # Init parameters.
        self.drop_path_ = DropPathPC(p_drop_prob)
        self.gamma_ = torch.nn.Parameter(p_init_gamma * torch.ones((1, p_num_features)))


    def forward(self, p_x, p_y, p_pc):
        """Forward method.

        Args:
            p_x (tensor): Input feature tensor.
            p_y (tensor): Input skip feature tensor.
            p_pc (Pointcloud): Input point cloud.

        Return:
            out_x (tensor): Output tensor.
        """

        return self.drop_path_(p_x*self.gamma_, p_pc) + p_y