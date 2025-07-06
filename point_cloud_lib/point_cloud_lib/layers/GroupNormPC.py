from abc import ABC, abstractmethod
import torch

from torch_scatter import scatter_mean

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.layers import NormLayerPC

class GroupNormPC(NormLayerPC):
    """Class to represent a normalization layer.
    """

    def __init__(self, p_num_features, p_num_groups=8):
        """Constructor.

        Args:
            p_num_features (int): Number of features.
        """

        # Super class init.
        super(GroupNormPC, self).__init__(p_num_features)

        # Save parameters.
        self.num_groups_ = p_num_groups

        # Create the parameters.
        self.gamma_ = torch.nn.Parameter(torch.empty(1, p_num_features))
        self.gamma_.data.fill_(1.0)

        self.betas_ = torch.nn.Parameter(torch.empty(1, p_num_features))
        self.betas_.data.fill_(0.0) 


    def forward(self, p_x, p_pc):
        """Forward method.

        Args:
            p_x (tensor): Input tensor.
            p_pc (Pointcloud): Input point cloud.
        """
        eps_val = 1e-8
        group_size = self.num_features_//self.num_groups_

        cur_x = p_x.reshape((-1, self.num_groups_))
        cur_batch_ids = p_pc.batch_ids_.reshape((-1, 1)).\
            repeat(1, group_size).to(torch.int64).reshape((-1,))

        feat_batch_means = scatter_mean(cur_x, cur_batch_ids, dim=0)
        feat_means = torch.index_select(feat_batch_means, 0, cur_batch_ids)

        feat_stddevs = (cur_x - feat_means)**2
        feat_batch_stddevs = scatter_mean(feat_stddevs, cur_batch_ids, dim=0)
        feat_stddevs = torch.index_select(feat_batch_stddevs, 0, cur_batch_ids)

        cur_x = (cur_x - feat_means)/torch.sqrt(feat_stddevs + eps_val)
        cur_x = cur_x.reshape((-1, self.num_features_))
        return cur_x*self.gamma_ + self.betas_