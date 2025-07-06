import numpy as np
import torch
import point_cloud_lib as pclib

from .Decoder import Decoder
from .PatchDecoder import PatchDecoder

class FPNDecoder(pclib.layers.PreProcessModule):
    """FPN Encoder network.
    """

    def __init__(self, 
                 p_encoder_feats,
                 p_fpn_dec_feats,
                 p_patch_num_levels,
                 p_conv_factory,
                 p_norm_layer,
                 p_neigh_type,
                 p_radius_scale,
                 p_num_knn,
                 p_max_path_drop):
        """Constructor.

        Args:
            p_encoder_feats (list int): Encoder feature sizes.
            p_fpn_dec_feats (int): Number of features in the fpn decoder.
            p_patch_num_levels (int): Number of levels patch decoder.
            p_conv_factory (IConvLayerFactory): Convolution factory.
            p_norm_layer (NormLayerPC): Normalization layer.
            p_neigh_type (string): Neighborhood type.
            p_radius_scale (float): Radius scale used in BQ.
            p_num_knn (int): Number of neigbors in kNN.
            p_max_path_drop (float): Maximum drop out path.
        """
        
        # Super class init.
        super(FPNDecoder, self).__init__()

        # Save parameters.
        self.encoder_feats_ = p_encoder_feats
        self.fpn_dec_feats_ = p_fpn_dec_feats
        self.conv_factory_ = p_conv_factory
        self.norm_layer_ = p_norm_layer
        self.neigh_type_ = p_neigh_type
        self.radius_scale_ = p_radius_scale
        self.num_knn_ = p_num_knn
        self.max_path_drop_ = p_max_path_drop

        # Create standard decoder.
        self.DECODER_ = Decoder(
            p_encoder_feats=p_encoder_feats,
            p_conv_factory=self.conv_factory_,
            p_norm_layer=p_norm_layer,
            p_neigh_type=p_neigh_type,
            p_radius_scale=p_radius_scale,
            p_num_knn=p_num_knn,
            p_max_path_drop=p_max_path_drop)
        
        # Patch decoder.
        self.PATCH_DECODER_ = PatchDecoder(
            p_num_feats=p_fpn_dec_feats,
            p_num_levels = p_patch_num_levels,
            p_conv_fatory = self.conv_factory_,
            p_norm_layer = p_norm_layer,
            p_neigh_type = p_neigh_type,
            p_radius_scale = p_radius_scale,
            p_num_knn = p_num_knn)
        
        # Create layers.
        self.BN_LAYERS_ = torch.nn.ModuleList()
        self.CONV_LAYERS_ = torch.nn.ModuleList()
        self.LINEAR_LAYERS_ = torch.nn.ModuleList()
        for cur_feats in reversed(self.encoder_feats_[1:]):

            self.BN_LAYERS_.append(p_norm_layer(cur_feats))
            self.LINEAR_LAYERS_.append(torch.nn.Linear(
                cur_feats, p_fpn_dec_feats))
            self.CONV_LAYERS_.append(self.conv_factory_.create_conv_layer(
                p_fpn_dec_feats, p_fpn_dec_feats))
            self.BN_LAYERS_.append(p_norm_layer(p_fpn_dec_feats))
            
        self.LINEAR_LAYERS_.append(torch.nn.Linear(
            self.encoder_feats_[0], p_fpn_dec_feats))
        self.BN_LAYERS_.append(p_norm_layer(p_fpn_dec_feats))
            

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

        # Standard decoder.
        dec_feats = self.DECODER_(p_hierarchy, p_in_feats, p_levels_radii)

        # Compute neighborhoods.
        with torch.no_grad():
            neighs = []
            last_level = len(p_hierarchy.pcs_)-1
            dest_level = last_level-len(p_in_feats)+1
            for cur_iter in range(len(self.CONV_LAYERS_)):
                cur_radius = self.radius_scale_*p_levels_radii[last_level-cur_iter]
                neighs.append(p_hierarchy.create_neighborhood(
                    last_level-cur_iter, dest_level, 
                    p_neigh_method = self.neigh_type_, 
                    bq_radius = cur_radius, neihg_k = self.num_knn_))

        # Upsample features.
        x = self.LINEAR_LAYERS_[-1](dec_feats[-1])
        x = self.BN_LAYERS_[-1](x, p_hierarchy.pcs_[dest_level])
        for cur_iter in range(len(self.CONV_LAYERS_)):
            cur_level = last_level-cur_iter

            cur_x = self.BN_LAYERS_[cur_iter*2](
                dec_feats[cur_iter], p_hierarchy.pcs_[cur_level])
            cur_x = self.LINEAR_LAYERS_[cur_iter](cur_x)
            cur_x = self.CONV_LAYERS_[cur_iter](
                p_pc_in = p_hierarchy.pcs_[cur_level],
                p_pc_out = p_hierarchy.pcs_[dest_level],
                p_in_features = cur_x,
                p_neighborhood = neighs[cur_iter])
            cur_x = self.BN_LAYERS_[cur_iter*2+1](
                cur_x, p_hierarchy.pcs_[dest_level])
            
            x = x + cur_x

        # Upsample patch.
        x = self.PATCH_DECODER_(p_hierarchy, x, p_levels_radii)

        return x
            