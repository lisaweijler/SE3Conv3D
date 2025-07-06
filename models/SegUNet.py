from abc import abstractmethod
import numpy as np
import torch
import point_cloud_lib as pclib

from .Encoder import Encoder
from .Decoder import Decoder

class SegUNet(pclib.layers.PreProcessModule):
    """Segmentation network.
    """

    # Encoder
    PATCH_NUM_LEVELS = 1
    PATCH_NUM_FEATURES = [8, 16]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 2, 2, 2, 2]
    NUM_FEATURES = [64, 128, 192, 256, 320]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0
    SEG_HEAD_FEATS = 128

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
        super(SegUNet, self).__init__()

        # Create convolution factory.
        self.conv_factory_ = self.__create_conv_factory__()

        # Create encoder.
        self.ENCODER_ = Encoder(
            p_num_in_feats=p_num_in_feats,
            p_patch_num_levels=self.PATCH_NUM_LEVELS,
            p_patch_num_features=self.PATCH_NUM_FEATURES,
            p_patch_norm_layer=self.PATCH_NORM_LAYER,
            p_patch_neigh_type=self.PATH_NEIGH_TYPE,
            p_patch_radius_scale=self.PATCH_RADIUS_SCALE,
            p_patch_num_knn=self.PATCH_NUM_KNN,
            p_conv_factory=self.conv_factory_,
            p_conv_factory_blocks=self.conv_factory_,
            p_block_layer=self.BLOCK_LAYER,
            p_norm_layer=self.NORM_LAYER,
            p_num_blocks=self.NUM_BLOCKS,
            p_num_features=self.NUM_FEATURES,
            p_neigh_type=self.NEIGH_TYPE,
            p_radius_scale=self.RADIUS_SCALE,
            p_num_knn=self.NUM_KNN,
            p_radius_scale_blocks=self.RADIUS_SCALE,
            p_num_knn_blocks=self.NUM_KNN,
            p_max_path_drop=p_max_path_drop) 
        
        # Create decoder.
        self.DECODER_ = Decoder(
            p_encoder_feats=self.NUM_FEATURES,
            p_conv_factory=self.conv_factory_,
            p_norm_layer=self.NORM_LAYER,
            p_neigh_type=self.NEIGH_TYPE,
            p_radius_scale=self.RADIUS_SCALE,
            p_num_knn=self.NUM_KNN,
            p_max_path_drop=p_max_path_dec_drop)

        # Segmentation head.
        self.SEG_BN_ = self.NORM_LAYER(self.NUM_FEATURES[0])
        self.SEG_CONV_ = self.conv_factory_.create_conv_layer(
            self.NUM_FEATURES[0], self.SEG_HEAD_FEATS)
        self.SEG_BN_2_ = self.NORM_LAYER(self.SEG_HEAD_FEATS)
        self.ACT_FUNCT_ = torch.nn.GELU()
        self.SEG_LINEAR_ = torch.nn.Linear(self.SEG_HEAD_FEATS, p_num_out_classes)


    @abstractmethod
    def __create_conv_factory__(self):
        """Abstract method to create the convolution factory.

        Returns:
            (ConvFactory): Convolution factory.
        """
        pass


    def forward(self, p_hierarchy, p_in_feats, p_levels_radii, p_out_pc):
        """Forward method.

        Args:
            p_hierarchy (PointHierarchy): Hierarchy of point clouds.
            p_in_feats (tensor): Input point features.
            p_levels_radii (list float): List of radii for each level 
                in the hierarcy.
            p_out_pc (PointCloud): Output point cloud.

        Returns:
            (tensor): Model output.            
        """

        # Create neighborhoods.
        with torch.no_grad():
            if self.NEIGH_TYPE == "knn":
                neigh_out = pclib.pc.KnnNeighborhood(
                        p_hierarchy.pcs_[self.PATCH_NUM_LEVELS], 
                        p_out_pc, self.DEC_NEIGH_K)
            elif self.NEIGH_TYPE == "ball_query":    
                neigh_out = pclib.pc.BQNeighborhood(
                        p_hierarchy.pcs_[self.PATCH_NUM_LEVELS], 
                        p_out_pc, self.RADIUS_SCALE*p_levels_radii[self.PATCH_NUM_LEVELS])
        
        # Process encoder.
        feats = self.ENCODER_(p_hierarchy, p_in_feats, p_levels_radii)

        # Process decoder.
        x = self.DECODER_(p_hierarchy, feats, p_levels_radii)
        x = x[-1]

        # Segmentation head.
        x = self.SEG_BN_(x, p_hierarchy.pcs_[self.PATCH_NUM_LEVELS])
        x = self.SEG_CONV_(
            p_pc_in = p_hierarchy.pcs_[self.PATCH_NUM_LEVELS],
            p_pc_out = p_out_pc,
            p_in_features = x,
            p_neighborhood = neigh_out)
        x = self.SEG_BN_2_(x, p_out_pc)
        x = self.ACT_FUNCT_(x)
        x = self.SEG_LINEAR_(x)

        return x


class SegUNetKPGauss(SegUNet):
    def __create_conv_factory__(self):
        return pclib.layers.PNEConvLayerFactory(
            p_dims = 3,
            p_num_basis = 16,
            p_pne_type = "kp_gauss")
class SegUNetLoRaFull(SegUNet):
    def __create_conv_factory__(self):
        return pclib.layers.LoRaConvLayerFactory(
            p_dims = 3,
            p_num_basis = 16,
            p_rank_type = "full")
class SegUNetLoRaPlaneVec(SegUNet):
    def __create_conv_factory__(self):
        return pclib.layers.LoRaConvLayerFactory(
            p_dims = 3,
            p_num_basis = 16,
            p_rank_type = "plane_vec")
class SegUNetLoRaVec(SegUNet):
    def __create_conv_factory__(self):
        return pclib.layers.LoRaConvLayerFactory(
            p_dims = 3,
            p_num_basis = 16,
            p_rank_type = "vec")

class SegUNet10(SegUNet):

    PATCH_NUM_LEVELS = 1
    PATCH_NUM_FEATURES = [8, 16]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 2, 2, 2, 2]
    NUM_FEATURES = [64, 128, 192, 256, 320]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0
    SEG_HEAD_FEATS = 128


class SegUNetKPGauss10(SegUNet10,SegUNetKPGauss):
    pass
class SegUNetLoRaFull10(SegUNet10,SegUNetLoRaFull):
    pass
class SegUNetLoRaPlaneVec10(SegUNet10,SegUNetLoRaPlaneVec):
    pass
class SegUNetLoRaVec10(SegUNet10,SegUNetLoRaVec):
    pass