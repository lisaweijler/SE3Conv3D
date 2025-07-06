from abc import abstractmethod
import numpy as np
import torch
import point_cloud_lib as pclib

from .Encoder import Encoder

class ClassNet(pclib.layers.PreProcessModule):
    """Classification network.
    """

    PATCH_NUM_LEVELS = 1
    PATCH_NUM_FEATURES = [8, 16]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 2, 2, 2, 2]
    NUM_FEATURES = [32, 64, 128, 256, 512]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0
    POOLING_METHOD = "avg"
    G_EQUIV_FEATURE_POOLING_METHOD = None
    GLOBAL_EQUIV_FEATUREVECTOR = False


    def __init__(self, 
                 p_num_in_feats,
                 p_num_out_classes,
                 p_max_path_drop = 0.2):
        """Constructor.

        Args:
            p_num_in_feats (int): Number input features.
            p_num_out_classes (int): Number output features.
            p_max_path_drop (float): Maximum drop out path.
        """
        
        # Super class init.
        super(ClassNet, self).__init__()

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
        


        self.CLASS_BN_ = torch.nn.BatchNorm1d(self.NUM_FEATURES[-1])
        self.CLASSHEAD_ = torch.nn.Linear(self.NUM_FEATURES[-1], p_num_out_classes)

        if self.GLOBAL_EQUIV_FEATUREVECTOR:
            self.ALMOST_LAST_BN_ = self.ENCODER_.norm_layer_(self.ENCODER_.num_features_[-1])
            self.GLOBAL_CONV_DOWN_ = self.ENCODER_.conv_factory_.create_conv_layer(
                    self.ENCODER_.num_features_[-1], self.ENCODER_.num_features_[-1]*2)
            
            self.LAST_BN_ = torch.nn.BatchNorm1d(self.ENCODER_.num_features_[-1]*2)
            self.LAST_LINEAR_ = torch.nn.Linear(self.ENCODER_.num_features_[-1]*2, self.ENCODER_.num_features_[-1]*2)
           


    @abstractmethod
    def __create_conv_factory__(self):
        """Abstract method to create the convolution factory.

        Returns:
            (ConvFactory): Convolution factory.
        """
        pass


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
        
        # Process encoder.
        feats = self.ENCODER_(p_hierarchy, p_in_feats, p_levels_radii)



        # Classification - need inviariant features        
        if not self.GLOBAL_EQUIV_FEATUREVECTOR:
            # if group conv - pool features to invariant features then global pool.
            if self.G_EQUIV_FEATURE_POOLING_METHOD is not None:
                x = p_hierarchy.pcs_[-1].global_pooling_specific_feature_pooling(feats[-1],
                                                                                p_global_pooling_method = self.POOLING_METHOD, 
                                                                                p_feature_pooling_method = self.G_EQUIV_FEATURE_POOLING_METHOD)

        

            # Global mean.
            else:
                x = p_hierarchy.pcs_[-1].global_pooling(feats[-1], p_pooling_method = self.POOLING_METHOD)
            
            # Classification head.
            x = self.CLASS_BN_(x)
            x = self.CLASSHEAD_(x)

        else:
            # last convolution
            # Create downsample.
        


            x = self.ALMOST_LAST_BN_(feats[-1], p_hierarchy.pcs_[-2])
            x = self.GLOBAL_CONV_DOWN_(
                    p_pc_in = p_hierarchy.pcs_[-2],
                    p_pc_out = p_hierarchy.pcs_[-1],
                    p_in_features = x,
                    p_neighborhood = p_hierarchy.create_neighborhood(
                    len(p_hierarchy.pcs_)-2, 
                    len(p_hierarchy.pcs_)-1, 
                    p_neigh_method = "knn", 
                     neihg_k = p_hierarchy.pcs_[-2].pts_.shape[0]/p_hierarchy.pcs_[-2].batch_size_)) # all points
            x = self.LAST_BN_(x)
            x = self.LAST_LINEAR_(x)


        

        return x
