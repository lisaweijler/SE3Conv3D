import torch

from point_cloud_lib.layers import SkipConnection, PreProcessModule

class Block(PreProcessModule):
    """ResNet Bottleneck block.
    """

    def __init__(self, 
        p_in_features, 
        p_out_features, 
        p_conv_fact,
        p_norm_layer,
        p_path_drop_prob):
        """Constructor.
        
        Args:
            p_in_features (int) 
            p_out_features (int)
            p_conv_fact (Conv factory)
            p_norm_layer (Normalization layer)
            p_path_drop_prob (float): Path drop probability.
        """

        # Super class init.
        super(Block, self).__init__()

        # Save sizes.
        self.feat_input_size_ = p_in_features
        self.feat_output_size_ = p_out_features


    def forward(self, 
        p_pc_in,
        p_in_features,
        p_neighborhood):
        """Forward method.
        
        Args:
            p_pc_in (Pointcloud): Input point cloud.
            p_in_features (tensor): Input features.
            p_neighborhood (Neighborhood): Neighborhood.
        Returns:
            (tensor) Output features.
        """
        pass