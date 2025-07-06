import os
import sys
import torch
import point_cloud_lib as pclib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(TASK_DIR)
sys.path.append(ROOT_DIR)

from models import FPNSegUNet

########## SEGMENTATION


class FPNSegUNetFAUST(FPNSegUNet):

    PATCH_NUM_LEVELS = 1
    PATCH_NUM_FEATURES = [32]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATCH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 2, 2, 2]
    NUM_FEATURES = [32, 64, 128, 256]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0
    RADIUS_SCALE_DEC = 2.0
    NUM_KNN_DEC = 0
    RADIUS_SCALE_BLOCKS = 2.0
    NUM_KNN_BLOCKS = 0
    FPN_DEC_FEATS = 32
    NUM_HIDDEN_SEG_HEAD = 0


class FPNSegUNetScanNet(FPNSegUNet):

    PATCH_NUM_LEVELS = 0  # 2#0
    PATCH_NUM_FEATURES = []  # [16,32]#[]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATCH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 3, 4, 6, 4]  # [2,2,2,2,2]
    NUM_FEATURES = [64, 128, 192, 256, 320]  # [128, 192, 256, 320, 384]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0
    RADIUS_SCALE_DEC = 2.0
    NUM_KNN_DEC = 0
    RADIUS_SCALE_BLOCKS = 2.0
    NUM_KNN_BLOCKS = 0
    FPN_DEC_FEATS = 128  # 96 # 128
    NUM_HIDDEN_SEG_HEAD = 0  # 2


class FPNSegUNetMLPGelu(FPNSegUNet):
    def __create_conv_factory__(self):
        return pclib.layers.PNEConvLayerFactory(
            p_dims=3, p_num_basis=32, p_pne_type="mlp_gelu"
        )

    def __create_conv_factory_blocks__(self):
        return self.__create_conv_factory__()


class FPNSegUNetMLPGeluRotEq(FPNSegUNet):
    def __create_conv_factory__(self):
        return pclib.layers.PNEConvLayerRotEquivFactory(
            p_dims=9, p_num_basis=32, p_pne_type="mlp_gelu"
        )

    def __create_conv_factory_blocks__(self):
        return self.__create_conv_factory__()


class FPNSegUNetMLPGeluScanNet(FPNSegUNetScanNet, FPNSegUNetMLPGelu):
    pass


class FPNSegUNetMLPGeluFAUST(FPNSegUNetFAUST, FPNSegUNetMLPGelu):
    pass


class FPNSegUNetMLPGeluRotEqScanNet(FPNSegUNetScanNet, FPNSegUNetMLPGeluRotEq):
    def forward(self, *args):
        pclib.layers.PNEConvLayerRotEquiv.empty_rot_tenors_cache()
        x = super().forward(*args)
        return args[-1].feature_pooling(x, p_pooling_method="avg")


class FPNSegUNetMLPGeluRotEqFAUST(FPNSegUNetFAUST, FPNSegUNetMLPGeluRotEq):
    def forward(self, *args):
        pclib.layers.PNEConvLayerRotEquiv.empty_rot_tenors_cache()
        x = super().forward(*args)
        return args[-1].feature_pooling(x, p_pooling_method="avg")


class FPNSegUNetMLPGeluRotEqFAUSTmax(FPNSegUNetFAUST, FPNSegUNetMLPGeluRotEq):
    def forward(self, *args):
        pclib.layers.PNEConvLayerRotEquiv.empty_rot_tenors_cache()
        x = super().forward(*args)
        return args[-1].feature_pooling(x, p_pooling_method="max")
