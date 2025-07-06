import torch

from point_cloud_lib.layers import SkipConnection, Block

class ResNetFormer(Block):
    """ResNetFormer block.
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
        super(ResNetFormer, self).__init__(
            p_in_features, 
            p_out_features, 
            p_conv_fact,
            p_norm_layer,
            p_path_drop_prob)

        # Create the activation function.
        self.act_func_ = torch.nn.GELU()
        
        # Create the residual path.
        self.feat_scale_factor_ = 2
        self.spatial_conv_ = p_conv_fact.create_conv_layer( 
            self.feat_input_size_, self.feat_input_size_)
        self.norm_1_ = p_norm_layer(self.feat_input_size_)
        self.norm_2_ = p_norm_layer(self.feat_input_size_)
        self.linear_1_ = torch.nn.Linear(self.feat_input_size_, self.feat_input_size_*self.feat_scale_factor_) 
        self.linear_2_ = torch.nn.Linear(self.feat_input_size_*self.feat_scale_factor_, self.feat_output_size_)
        
        self.skip_path_1_ = SkipConnection(p_path_drop_prob, self.feat_input_size_)
        self.skip_path_2_ = SkipConnection(p_path_drop_prob, self.feat_output_size_)

        if self.feat_input_size_ != self.feat_output_size_:
            self.skip_conv_ = torch.nn.Linear(self.feat_input_size_, self.feat_output_size_)


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

        # Residual path 1.
        x = p_in_features
        x = self.norm_1_(x, p_pc_in)
        x = self.spatial_conv_(
            p_pc_in = p_pc_in,
            p_pc_out = p_pc_in,
            p_in_features = x,
            p_neighborhood = p_neighborhood)
        
        # Skip connection 1.
        x = self.skip_path_1_(x, p_in_features, p_pc_in)

        # Residual path 2.
        y = self.norm_2_(x, p_pc_in)
        y = self.linear_1_(y)
        y = self.act_func_(y)
        y = self.linear_2_(y)

        # Skip path.
        if self.feat_input_size_ != self.feat_output_size_:
            skip_path = self.skip_conv_(x)
        else:
            skip_path = x
        
        # Skip connection.
        return self.skip_path_2_(y, skip_path, p_pc_in)