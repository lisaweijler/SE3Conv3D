import numpy as np
import torch
import point_cloud_lib as pclib

class Decoder(pclib.layers.PreProcessModule):
    """Encoder network.
    """

    def __init__(self, 
                 p_encoder_feats,
                 p_conv_factory,
                 p_norm_layer,
                 p_neigh_type,
                 p_radius_scale,
                 p_num_knn,
                 p_max_path_drop):
        """Constructor.

        Args:
            p_encoder_feats (list int): Encoder feature sizes.
            p_conv_factory (IConvLayerFactory): Convolution factory.
            p_norm_layer (NormLayerPC): Normalization layer.
            p_neigh_type (string): Neighborhood type.
            p_radius_scale (float): Radius scale used in BQ.
            p_num_knn (int): Number of neigbors in kNN.
            p_max_path_drop (float): Maximum drop out path.
        """
        
        # Super class init.
        super(Decoder, self).__init__()

        # Save parameters.
        self.encoder_feats_ = p_encoder_feats
        self.conv_factory_ = p_conv_factory
        self.norm_layer_ = p_norm_layer
        self.neigh_type_ = p_neigh_type
        self.radius_scale_ = p_radius_scale
        self.num_knn_ = p_num_knn
        self.max_path_drop_ = p_max_path_drop

        # Create layers.
        drop_paths = np.linspace(self.max_path_drop_, 0, len(self.encoder_feats_)-1)
        self.BN_LAYERS_ = torch.nn.ModuleList()
        self.CONV_LAYERS_ = torch.nn.ModuleList()
        self.SKIP_LAYERS_ = torch.nn.ModuleList()
        for cur_level in reversed(range(len(self.encoder_feats_)-1)):

            self.BN_LAYERS_.append(p_norm_layer(self.encoder_feats_[cur_level+1]))
            self.CONV_LAYERS_.append(
                self.conv_factory_.create_conv_layer(
                    self.encoder_feats_[cur_level+1], 
                    self.encoder_feats_[cur_level]))
            self.SKIP_LAYERS_.append(pclib.layers.SkipConnection(
                drop_paths[cur_level], self.encoder_feats_[cur_level]))
            

    def forward(self, p_hierarchy, p_in_feats, p_levels_radii):
        """Forward method.

        Args:
            p_hierarchy (PointHierarchy): Hierarchy of point clouds.
            p_in_feats (list of tensor): Input list of point features.
            p_levels_radii (list float): List of radii for each level 
                in the hierarcy.

        Returns:
            (tensor): Model output.            
        """

        # Compute neighborhoods.
        last_level = len(p_hierarchy.pcs_)-1
        with torch.no_grad():
            neighs = []
            for cur_iter in range(len(self.CONV_LAYERS_)):
                cur_level = last_level-cur_iter
                cur_radius = self.radius_scale_*p_levels_radii[cur_level]
                neighs.append(p_hierarchy.create_neighborhood(
                    cur_level, cur_level-1, 
                    p_neigh_method = self.neigh_type_, 
                    bq_radius = cur_radius, neihg_k = self.num_knn_))

        # Upsample features.
        in_feat_rev = [p_in_feats[i] for i in reversed(range(len(p_in_feats)))]
        x = in_feat_rev[0]
        out_features = [x]
        for cur_iter in range(len(self.CONV_LAYERS_)):
            cur_level = last_level-cur_iter

            x = self.BN_LAYERS_[cur_iter](x, p_hierarchy.pcs_[cur_level])
            x = self.CONV_LAYERS_[cur_iter](
                p_pc_in = p_hierarchy.pcs_[cur_level],
                p_pc_out = p_hierarchy.pcs_[cur_level-1],
                p_in_features = x,
                p_neighborhood = neighs[cur_iter])
            x = self.SKIP_LAYERS_[cur_iter](x, in_feat_rev[cur_iter+1], 
                p_hierarchy.pcs_[cur_level-1])
            
            out_features.append(x)

        return out_features
            