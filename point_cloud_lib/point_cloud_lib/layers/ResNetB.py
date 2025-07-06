import torch

from point_cloud_lib.layers import SkipConnection, Block

class ResNetB(Block):
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
        super(ResNetB, self).__init__(
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
            self.feat_input_size_//self.feat_scale_factor_, 
            self.feat_input_size_//self.feat_scale_factor_)
        self.norm_ = p_norm_layer(self.feat_input_size_)
        self.linear_1_ = torch.nn.Linear(self.feat_input_size_, self.feat_input_size_//self.feat_scale_factor_) 
        self.linear_2_ = torch.nn.Linear(self.feat_input_size_//self.feat_scale_factor_, self.feat_output_size_)
        
        self.skip_path_ = SkipConnection(p_path_drop_prob, self.feat_output_size_)

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

        # Residual path.
        x = p_in_features
        x = self.norm_(x, p_pc_in)
        x = self.linear_1_(x)
        x = self.spatial_conv_(
            p_pc_in = p_pc_in,
            p_pc_out = p_pc_in,
            p_in_features = x,
            p_neighborhood = p_neighborhood)
        x = self.act_func_(x)
        x = self.linear_2_(x)

        # Skip path.
        if self.feat_input_size_ != self.feat_output_size_:
            skip_path = self.skip_conv_(p_in_features)
        else:
            skip_path = p_in_features
        
        # Skip connection.
        return self.skip_path_(x, skip_path, p_pc_in)