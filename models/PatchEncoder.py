import numpy as np
import torch
import point_cloud_lib as pclib

class PatchEncoder(pclib.layers.PreProcessModule):
    """Patch Encoder network.
    """

    def __init__(self, 
                 p_num_in_feats,
                 p_num_out_feats,
                 p_num_levels,
                 p_num_features,
                 p_conv_fatory,
                 p_norm_layer,
                 p_neigh_type,
                 p_radius_scale,
                 p_num_knn):
        """Constructor.

        Args:
            p_num_in_feats (int): Number input features.
            p_num_out_feats (int): Number output features.
            p_num_levels (int): Number of levels.
            p_num_features (list int): List of number features.
            p_conv_fatory (IConvLayerFactory): Convolution factory.
            p_norm_layer (NormLayerPC): Normalization layer.
            p_neigh_type (string): Type of neighborhood.
            p_radius_scale (float): Scale radius in BQ.
            p_num_knn (int): Number of neighbors in kNN.
        """
        
        # Super class init.
        super(PatchEncoder, self).__init__()

        # Save parameters.
        self.num_levels_ = p_num_levels
        self.neigh_type_ = p_neigh_type
        self.radius_scale_ = p_radius_scale
        self.num_knn_ = p_num_knn

        # Create layers.
        cur_features = p_num_in_feats
        self.BN_LAYERS_ = torch.nn.ModuleList()
        self.CONV_LAYERS_ = torch.nn.ModuleList()
        for cur_level in range(p_num_levels):

            self.CONV_LAYERS_.append(p_conv_fatory.create_conv_layer(
                cur_features, p_num_features[cur_level]))
            self.CONV_LAYERS_.append(p_conv_fatory.create_conv_layer(
                p_num_features[cur_level], p_num_features[cur_level]))
            
            self.BN_LAYERS_.append(p_norm_layer(p_num_features[cur_level]))
            self.BN_LAYERS_.append(p_norm_layer(p_num_features[cur_level]))

            cur_features = p_num_features[cur_level]

        self.LINEAR_ = torch.nn.Linear(cur_features, p_num_out_feats)
        
        self.BN_LAYERS_.append(p_norm_layer(p_num_out_feats))

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
                cur_radius = self.radius_scale_*p_levels_radii[cur_level]
                neighs.append(p_hierarchy.create_neighborhood(cur_level, cur_level+1, 
                    p_neigh_method = self.neigh_type_, 
                    bq_radius = cur_radius, neihg_k = self.num_knn_))
                cur_radius = self.radius_scale_*p_levels_radii[cur_level+1]
                neighs.append(p_hierarchy.create_neighborhood(cur_level+1, cur_level+1, 
                    p_neigh_method = self.neigh_type_, 
                    bq_radius = cur_radius, neihg_k = self.num_knn_))
            
        # Compute features.
        x = p_in_feats
        for cur_level in range(self.num_levels_):

            x = self.CONV_LAYERS_[cur_level*2](
                p_pc_in = p_hierarchy.pcs_[cur_level],
                p_pc_out = p_hierarchy.pcs_[cur_level+1],
                p_in_features = x,
                p_neighborhood = neighs[cur_level*2])
            x = self.BN_LAYERS_[cur_level*2](x, p_hierarchy.pcs_[cur_level+1])
            x = self.ACT_FUNCT_(x)
            x = self.CONV_LAYERS_[cur_level*2+1](
                p_pc_in = p_hierarchy.pcs_[cur_level+1],
                p_pc_out = p_hierarchy.pcs_[cur_level+1],
                p_in_features = x,
                p_neighborhood = neighs[cur_level*2+1])
            x = self.BN_LAYERS_[cur_level*2+1](x, p_hierarchy.pcs_[cur_level+1])
            x = self.ACT_FUNCT_(x)

        x = self.LINEAR_(x)
        x = self.BN_LAYERS_[self.num_levels_*2](x, p_hierarchy.pcs_[self.num_levels_])

        return x