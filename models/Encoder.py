import numpy as np
import torch
import point_cloud_lib as pclib

from .PatchEncoder import PatchEncoder

class Encoder(pclib.layers.PreProcessModule):
    """Encoder network.
    """

    def __init__(self, 
                 p_num_in_feats,
                 p_patch_num_levels,
                 p_patch_num_features,
                 p_patch_norm_layer,
                 p_patch_neigh_type,
                 p_patch_radius_scale,
                 p_patch_num_knn,
                 p_conv_factory,
                 p_conv_factory_blocks,
                 p_block_layer,
                 p_norm_layer,
                 p_num_blocks,
                 p_num_features,
                 p_neigh_type,
                 p_radius_scale,
                 p_num_knn,
                 p_radius_scale_blocks,
                 p_num_knn_blocks,
                 p_max_path_drop):
        """Constructor.

        Args:
            p_num_in_feats (int): Number input features.
            p_patch_num_levels (int): Number of levels in the patch embedding.
            p_patch_num_features (list int): List of features in the patch embedding.
            p_patch_norm_layer (NormLayerPC): Normalization layer.
            p_patch_neigh_type (string): Neighborhood type in the patch embedding.
            p_patch_radius_scale (float): Radius scale used in BQ in the patch embedding.
            p_patch_num_knn (int): Number of neighbors in kNN.
            p_conv_factory (IConvLayerFactory): Convolution factory.
            p_conv_factory_blocks (IConvLayerFactory): Convolution factory for the block layers.
            p_block_layer (Block): Block layer.
            p_norm_layer (NormLayerPC): Normalization layer.
            p_num_blocks (list int): Number of blocks per level.
            p_num_features (list int): Number of features in each block.
            p_neigh_type (string): Neighborhood type.
            p_radius_scale (float): Radius scale used in BQ.
            p_num_knn (int): Number of neigbors in kNN.
            p_radius_scale_blocks (float): Radius scale used in BQ inside the blocks.
            p_num_knn_blocks (int): Number of neigbors in kNN inside the blocks.
            p_max_path_drop (float): Maximum drop out path.
        """
        
        # Super class init.
        super(Encoder, self).__init__()

        # Save parameters.
        self.num_in_feats_ = p_num_in_feats
        self.patch_num_levels_ = p_patch_num_levels
        self.patch_num_features_ = p_patch_num_features
        self.patch_norm_layer_ = p_patch_norm_layer
        self.patch_neigh_type_ = p_patch_neigh_type
        self.patch_radius_scale_ = p_patch_radius_scale
        self.patch_num_knn_ = p_patch_num_knn
        self.conv_factory_ = p_conv_factory
        self.conv_factory_blocks_ = p_conv_factory_blocks
        self.block_layer_ = p_block_layer
        self.norm_layer_ = p_norm_layer
        self.num_blocks_ = p_num_blocks
        self.num_features_ = p_num_features
        self.neigh_type_ = p_neigh_type
        self.radius_scale_ = p_radius_scale
        self.num_knn_ = p_num_knn
        self.radius_scale_blocks_ = p_radius_scale_blocks
        self.num_knn_blocks_ = p_num_knn_blocks
        self.max_path_drop_ = p_max_path_drop

        # Create patch embedding.
        self.PATCH_EMB_ = PatchEncoder(
            p_num_in_feats = self.num_in_feats_,
            p_num_out_feats = self.num_features_[0],
            p_num_levels = self.patch_num_levels_,
            p_num_features = self.patch_num_features_,
            p_conv_fatory = self.conv_factory_,
            p_norm_layer = self.patch_norm_layer_,
            p_neigh_type = self.patch_neigh_type_,
            p_radius_scale = self.patch_radius_scale_,
            p_num_knn = self.patch_num_knn_)
        
        # Create blocks.
        drop_paths = np.linspace(0, self.max_path_drop_, np.sum(self.num_blocks_))
        self.BLOCKS_LIST_ = torch.nn.ModuleList()
        cur_block_id = 0
        for cur_num_feats, cur_num_blocks in zip(self.num_features_, self.num_blocks_):
            self.BLOCKS_LIST_.append(torch.nn.ModuleList([
                self.block_layer_(
                    p_in_features=cur_num_feats, 
                    p_out_features=cur_num_feats, 
                    p_conv_fact=self.conv_factory_blocks_,
                    p_norm_layer=self.norm_layer_,
                    p_path_drop_prob=drop_paths[cur_block_id+i])
                for i in range(cur_num_blocks)
                ]))
            cur_block_id += cur_num_blocks

        # Create downsample.
        self.BN_ = torch.nn.ModuleList()
        self.CONV_DOWN_ = torch.nn.ModuleList()
        for cur_level in range(len(self.num_features_)-1):
            self.BN_.append(self.norm_layer_(self.num_features_[cur_level]))
            self.CONV_DOWN_.append(self.conv_factory_.create_conv_layer(
                self.num_features_[cur_level], self.num_features_[cur_level+1]))


    def forward(self, p_hierarchy, p_in_feats, p_levels_radii):
        """Forward method.

        Args:
            p_hierarchy (PointHierarchy): Hierarchy of point clouds.
            p_in_feats (tensor): Input point features.
            p_levels_radii (list float): List of radii for each level 
                in the hierarcy.

        Returns:
            (list of tensor): Model output.            
        """

        # Patch embedding.
        x = p_in_feats
        x = self.PATCH_EMB_(p_hierarchy, x, p_levels_radii)
        
        # Compute neighborhoods.
        with torch.no_grad():
            neighs = []
            for cur_level in range(len(self.num_blocks_)-1):
                cur_radius = self.radius_scale_*p_levels_radii[cur_level+self.patch_num_levels_]
                cur_radius_block = self.radius_scale_blocks_*p_levels_radii[cur_level+self.patch_num_levels_]
                neighs.append(p_hierarchy.create_neighborhood(
                    cur_level+self.patch_num_levels_, 
                    cur_level+self.patch_num_levels_, 
                    p_neigh_method = self.neigh_type_, 
                    bq_radius = cur_radius_block, neihg_k = self.num_knn_blocks_))
                neighs.append(p_hierarchy.create_neighborhood(
                    cur_level+self.patch_num_levels_, 
                    cur_level+self.patch_num_levels_+1, 
                    p_neigh_method = self.neigh_type_, 
                    bq_radius = cur_radius, neihg_k = self.num_knn_))
            cur_radius_block = self.radius_scale_blocks_*p_levels_radii[self.patch_num_levels_+len(self.num_blocks_)-1]
            neighs.append(p_hierarchy.create_neighborhood(
                self.patch_num_levels_+len(self.num_blocks_)-1, 
                self.patch_num_levels_+len(self.num_blocks_)-1, 
                p_neigh_method = self.neigh_type_, 
                bq_radius = cur_radius_block, neihg_k = self.num_knn_blocks_))

        # Compute features.
        out_feat_list = []
        for cur_level in range(len(self.num_features_)):
            for cur_block in self.BLOCKS_LIST_[cur_level]:
                x = cur_block(
                    p_pc_in=p_hierarchy.pcs_[cur_level+self.patch_num_levels_],
                    p_in_features=x,
                    p_neighborhood=neighs[cur_level*2])
            out_feat_list.append(x)
            if cur_level < len(self.num_features_)-1:
                x = self.BN_[cur_level](x, p_hierarchy.pcs_[cur_level+self.patch_num_levels_])
                x = self.CONV_DOWN_[cur_level](
                    p_pc_in = p_hierarchy.pcs_[cur_level+self.patch_num_levels_],
                    p_pc_out = p_hierarchy.pcs_[cur_level+self.patch_num_levels_+1],
                    p_in_features = x,
                    p_neighborhood = neighs[cur_level*2+1])
                
        return out_feat_list
