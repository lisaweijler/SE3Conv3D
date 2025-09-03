import os
import sys
import torch
import point_cloud_lib as pclib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(TASK_DIR)
sys.path.append(ROOT_DIR)

from models import ClassNet


class ClassNet19Former(ClassNet):

    PATCH_NUM_LEVELS = 1
    PATCH_NUM_FEATURES = [32]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATCH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 3, 4, 6, 4]
    NUM_FEATURES = [
        32,
        64,
        128,
        256,
        512,
    ]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0


class ClassNet19FormerMax(ClassNet):

    PATCH_NUM_LEVELS = 1
    PATCH_NUM_FEATURES = [32]
    PATCH_NORM_LAYER = pclib.layers.BatchNormPC
    PATCH_NEIGH_TYPE = "ball_query"
    PATCH_RADIUS_SCALE = 2.0
    PATCH_NUM_KNN = 0
    BLOCK_LAYER = pclib.layers.ResNetFormer
    NORM_LAYER = pclib.layers.BatchNormPC
    NUM_BLOCKS = [2, 3, 4, 6, 4]
    NUM_FEATURES = [
        32,
        64,
        128,
        256,
        512,
    ]
    NEIGH_TYPE = "ball_query"
    RADIUS_SCALE = 2.0
    NUM_KNN = 0
    POOLING_METHOD = "avg"
    G_EQUIV_FEATURE_POOLING_METHOD = "max"


class ClassNetMLPGELU(ClassNet):
    def __create_conv_factory__(self):
        return pclib.layers.PNEConvLayerFactory(
            p_dims=3, p_num_basis=32, p_pne_type="mlp_gelu"
        )


class ClassNetRotEquivMLPGELU(ClassNet):
    def __create_conv_factory__(self):
        return pclib.layers.PNEConvLayerRotEquivFactory(
            p_dims=9, p_num_basis=32, p_pne_type="mlp_gelu"
        )


class ClassNetMLPGELU19Former(ClassNet19Former, ClassNetMLPGELU):
    pass


class ClassNetRotEquivMLPGELU19Former(ClassNet19Former, ClassNetRotEquivMLPGELU):
    def forward(self, *args):
        pclib.layers.PNEConvLayerRotEquiv.empty_rot_tenors_cache()
        return super().forward(*args)


class ClassNetRotEquivMLPGELU19FormerMax(ClassNet19FormerMax, ClassNetRotEquivMLPGELU):
    def forward(self, *args):
        pclib.layers.PNEConvLayerRotEquiv.empty_rot_tenors_cache()
        return super().forward(*args)
