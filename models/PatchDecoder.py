import numpy as np
import torch
import point_cloud_lib as pclib

class PatchDecoder(pclib.layers.PreProcessModule):
    """Patch Decoder network.
    """

    def __init__(self, 
                 p_num_feats,
                 p_num_levels,
                 p_conv_fatory,
                 p_norm_layer,
                 p_neigh_type,
                 p_radius_scale,
                 p_num_knn):
        """Constructor.

        Args:
            p_num_feats (int): Number of features.
            p_num_levels (int): Number of levels.
            p_conv_fatory (IConvLayerFactory): Convolution factory.
            p_norm_layer (NormLayerPC): Normalization layer.
            p_neigh_type (string): Type of neighborhood.
            p_radius_scale (float): Scale radius in BQ.
            p_num_knn (int): Number of neighbors in kNN.
        """
        
        # Super class init.
        super(PatchDecoder, self).__init__()

        # Save parameters.
        self.num_levels_ = p_num_levels
        self.neigh_type_ = p_neigh_type
        self.radius_scale_ = p_radius_scale
        self.num_knn_ = p_num_knn

        # Create layers.
        self.BN_LAYERS_ = torch.nn.ModuleList()
        self.CONV_LAYERS_ = torch.nn.ModuleList()
        for _ in range(p_num_levels):
            self.CONV_LAYERS_.append(p_conv_fatory.create_conv_layer(
                p_num_feats, p_num_feats))
            self.BN_LAYERS_.append(p_norm_layer(p_num_feats))

        self.ACT_FUNCT_ = torch.nn.GELU()

    
    def forward(self, p_hierarchy, p_in_feats, p_levels_radii):
        """Forward method.

        Args:
            p_hierarchy (PointHierarchy): Hierarchy of point clouds.
            p_in_feats (tensor): Input point features.
            p_levels_radii (list float): List of radii for each level 
                in the hierarcy.

        Returns:
            (tensor): Model output.            
        """

        # Create neighborhoods.
        with torch.no_grad():
            neighs = []
            for cur_level in range(self.num_levels_):
                cur_radius = self.radius_scale_*p_levels_radii[cur_level+1]
                neighs.append(p_hierarchy.create_neighborhood(
                    cur_level+1, cur_level, 
                    p_neigh_method = self.neigh_type_, 
                    bq_radius = cur_radius, neihg_k = self.num_knn_))

        # Compute features.
        x = p_in_feats
        for cur_level in reversed(range(self.num_levels_)):

            x = self.CONV_LAYERS_[cur_level](
                p_pc_in = p_hierarchy.pcs_[cur_level+1],
                p_pc_out = p_hierarchy.pcs_[cur_level],
                p_in_features = x,
                p_neighborhood = neighs[cur_level])
            x = self.BN_LAYERS_[cur_level](x, p_hierarchy.pcs_[cur_level])
            x = self.ACT_FUNCT_(x)

        return x