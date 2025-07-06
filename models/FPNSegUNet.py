from abc import abstractmethod
import numpy as np
import torch
import point_cloud_lib as pclib

from .Encoder import Encoder
from .FPNDecoder import FPNDecoder

class FPNSegUNet(pclib.layers.PreProcessModule):
    """Segmentation network.
    """

    # Encoder
    PATCH_NUM_LEVELS = 1
    PATCH_NUM_FEATURES = [8]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATCH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 2, 2, 2, 2]
    NUM_FEATURES = [64, 128, 192, 256, 320]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0
    RADIUS_SCALE_DEC = 1.5
    NUM_KNN_DEC = 0
    RADIUS_SCALE_BLOCKS = 2.0
    NUM_KNN_BLOCKS = 0
    FPN_DEC_FEATS = 128
    NUM_HIDDEN_SEG_HEAD = 0

    def __init__(self, 
                 p_num_in_feats,
                 p_num_out_classes,
                 p_max_path_drop = 0.2,
                 p_max_path_dec_drop = 0.0):
        """Constructor.

        Args:
            p_num_in_feats (int): Number input features.
            p_num_out_classes (int): Number output features.
            p_max_path_drop (float): Maximum drop out path.
            p_max_path_dec_drop (float): Maximum drop out path decoder.
        """
        
        # Super class init.
        super(FPNSegUNet, self).__init__()
        self.num_out_classes_ = p_num_out_classes

        # Create convolution factory.
        self.conv_factory_ = self.__create_conv_factory__()
        self.conv_factory_blocks_ = self.__create_conv_factory_blocks__()

        # Create encoder.
        self.ENCODER_ = Encoder(
            p_num_in_feats=p_num_in_feats,
            p_patch_num_levels=self.PATCH_NUM_LEVELS,
            p_patch_num_features=self.PATCH_NUM_FEATURES,
            p_patch_norm_layer=self.PATCH_NORM_LAYER,
            p_patch_neigh_type=self.PATCH_NEIGH_TYPE,
            p_patch_radius_scale=self.PATCH_RADIUS_SCALE,
            p_patch_num_knn=self.PATCH_NUM_KNN,
            p_conv_factory=self.conv_factory_,
            p_conv_factory_blocks=self.conv_factory_blocks_,
            p_block_layer=self.BLOCK_LAYER,
            p_norm_layer=self.NORM_LAYER,
            p_num_blocks=self.NUM_BLOCKS,
            p_num_features=self.NUM_FEATURES,
            p_neigh_type=self.NEIGH_TYPE,
            p_radius_scale=self.RADIUS_SCALE,
            p_num_knn=self.NUM_KNN,
            p_radius_scale_blocks=self.RADIUS_SCALE_BLOCKS,
            p_num_knn_blocks=self.NUM_KNN_BLOCKS,
            p_max_path_drop=p_max_path_drop) 
        
        # Create decoder.
        self.DECODER_ = FPNDecoder(
            p_encoder_feats=self.NUM_FEATURES,
            p_fpn_dec_feats=self.FPN_DEC_FEATS,
            p_patch_num_levels=self.PATCH_NUM_LEVELS,
            p_conv_factory=self.conv_factory_,
            p_norm_layer=self.NORM_LAYER,
            p_neigh_type=self.NEIGH_TYPE,
            p_radius_scale=self.RADIUS_SCALE_DEC,
            p_num_knn=self.NUM_KNN_DEC,
            p_max_path_drop=p_max_path_dec_drop)

        # Segmentation head.
        self.SEG_CONV_ = self.conv_factory_.create_conv_layer(
            self.FPN_DEC_FEATS, self.FPN_DEC_FEATS)
        self.HIDDEN_SEG_BN_LAYERS_ = torch.nn.ModuleList()
        self.HIDDEN_SEG_LINEAR_ = torch.nn.ModuleList()
        for _ in range(self.NUM_HIDDEN_SEG_HEAD):
            self.HIDDEN_SEG_BN_LAYERS_.append(
                self.NORM_LAYER(self.FPN_DEC_FEATS))
            self.HIDDEN_SEG_LINEAR_.append(
                torch.nn.Linear(self.FPN_DEC_FEATS, self.FPN_DEC_FEATS))
        self.SEG_BN_ = self.NORM_LAYER(self.FPN_DEC_FEATS)
        self.ACT_FUNCT_ = torch.nn.GELU()
        self.SEG_LINEAR_ = torch.nn.Linear(self.FPN_DEC_FEATS, p_num_out_classes)


    @abstractmethod
    def __create_conv_factory__(self):
        """Abstract method to create the convolution factory.

        Returns:
            (ConvFactory): Convolution factory.
        """
        pass

    @abstractmethod
    def __create_conv_factory_blocks__(self):
        """Abstract method to create the convolution factory for the block layers..

        Returns:
            (ConvFactory): Convolution factory.
        """
        pass


    def process_encoder_decoder(self,
                                p_hierarchy, 
                                p_in_feats, 
                                p_levels_radii):
        """Forward encoder-decoder method.

        Args:
            p_hierarchy (PointHierarchy): Hierarchy of point clouds.
            p_in_feats (tensor): Input point features.
            p_levels_radii (list float): List of radii for each level 
                in the hierarcy.
        Returns:
            (tensor): Model output.            
        """
        # Process encoder.
        feats = self.ENCODER_(p_hierarchy, p_in_feats, p_levels_radii)

        # Process decoder.
        x = self.DECODER_(p_hierarchy, feats, p_levels_radii)

        return x
    

    def process_last_upsample(self,
                            p_hierarchy, 
                            p_in_feats, 
                            p_levels_radii,
                            p_out_pc,
                            p_return_hidden = False):
        """Forward last upsample method.

        Args:
            p_hierarchy (PointHierarchy): Hierarchy of point clouds.
            p_in_feats (tensor): Input point features.
            p_levels_radii (list float): List of radii for each level 
                in the hierarcy.
            p_out_pc (PointCloud): Output point cloud.
            p_return_hidden (bool): If true, hidden features are returned.
        Returns:
            (tensor): Model output.            
        """

        # Create neighborhoods.
        with torch.no_grad():
            if self.NEIGH_TYPE == "knn":
                neigh_out = pclib.pc.KnnNeighborhood(
                        p_hierarchy.pcs_[0], 
                        p_out_pc, self.NUM_KNN)
            elif self.NEIGH_TYPE == "ball_query":    
                neigh_out = pclib.pc.BQNeighborhood(
                        p_hierarchy.pcs_[0], 
                        p_out_pc, self.RADIUS_SCALE*p_levels_radii[0])

        # Segmentation head.
        hidden = self.SEG_CONV_(
            p_pc_in = p_hierarchy.pcs_[0],
            p_pc_out = p_out_pc,
            p_in_features = p_in_feats,
            p_neighborhood = neigh_out)
        x = hidden
        for cur_iter in range(self.NUM_HIDDEN_SEG_HEAD):
            x = self.HIDDEN_SEG_BN_LAYERS_[cur_iter](x, p_out_pc)
            x = self.ACT_FUNCT_(x)
            x = self.HIDDEN_SEG_LINEAR_[cur_iter](x)
        x = self.SEG_BN_(x, p_out_pc)
        x = self.ACT_FUNCT_(x)
        x = self.SEG_LINEAR_(x)
        
        if p_return_hidden:
            return x, hidden
        else:
            return x


    def forward(self, 
                p_hierarchy, 
                p_in_feats, 
                p_levels_radii, 
                p_out_pc,
                p_return_hidden = False):
        """Forward method.

        Args:
            p_hierarchy (PointHierarchy): Hierarchy of point clouds.
            p_in_feats (tensor): Input point features.
            p_levels_radii (list float): List of radii for each level 
                in the hierarcy.
            p_out_pc (PointCloud): Output point cloud.
            p_return_hidden (bool): If true, hidden features are returned.

        Returns:
            (tensor): Model output.            
        """

        # Process encoder-decoder.
        x = self.process_encoder_decoder(p_hierarchy, p_in_feats, p_levels_radii)

        # Process last upsample.
        return self.process_last_upsample(p_hierarchy, x, p_levels_radii, 
            p_out_pc, p_return_hidden)
