"""This file contains the definition of different heterogeneous datasets used for training."""

import joblib
import numpy as np
import torch
import os
import h5py
from pathlib import Path
from torch.utils.data import Dataset, Sampler

import point_cloud_lib as pclib

MN40_BASE_AUGMENTATIONS = [
    {"name": "CenterAug", "p_apply_extra_tensors": [False]},
    {
        "name": "RotationAug",
        "p_prob": 1.0,
        "p_axis": 0,
        "p_min_angle": -np.pi / 24.0,
        "p_max_angle": np.pi / 24.0,
        "p_apply_extra_tensors": [True],
    },
    {
        "name": "RotationAug",
        "p_prob": 1.0,
        "p_axis": 2,
        "p_min_angle": -np.pi / 24.0,
        "p_max_angle": np.pi / 24.0,
        "p_apply_extra_tensors": [True],
    },
    {
        "name": "NoiseAug",
        "p_prob": 1.0,
        "p_stddev": 0.01,
        "p_apply_extra_tensors": [False],
    },
    {
        "name": "LinearAug",
        "p_prob": 1.0,
        "p_min_a": 0.9,
        "p_max_a": 1.1,
        "p_min_b": 0.0,
        "p_max_b": 0.0,
        "p_channel_independent": True,
        "p_apply_extra_tensors": [False],
    },
    {
        "name": "MirrorAug",
        "p_prob": 1.0,
        "p_mirror_prob": 0.5,
        "p_axes": [True, False, True],
        "p_apply_extra_tensors": [True],
    },
]


class DFaust_Collate:

    @staticmethod
    def collate(p_batch):
        batch_pts = []
        batch_ids = []
        batch_labels = []
        batch_features = []
        obj_ids = []
        for cur_iter, cur_batch in enumerate(p_batch):
            batch_pts.append(cur_batch[0])
            batch_ids.append(
                torch.ones(cur_batch[0].shape[0], dtype=torch.int32) * cur_iter
            )
            batch_labels.append(cur_batch[1])
            batch_features.append(cur_batch[2])
            obj_ids.append(cur_batch[3])

        batch_pts = torch.cat(batch_pts, 0).to(torch.float32)
        batch_features = torch.cat(batch_features, 0)
        batch_ids = torch.cat(batch_ids, 0).to(torch.int32)
        batch_labels = torch.cat(batch_labels, 0).to(torch.int64)

        return batch_pts, batch_features, batch_ids, batch_labels, obj_ids


class DFaustDS(Dataset):
    """DFaust data set as subset from AMASS dataset."""

    def __init__(
        self,
        p_data_folder,
        p_augmentation_cfg,
        p_num_pts=1024,
        p_split="train",
    ):
        """Constructor.

        Args:
            p_data_folder (string): Data folder path.
            p_augmentation_cfg (list of dict): List of dictionaries with
                the different configurations for the data augmentation
                techniques.
            p_num_pts (int): Number of points.
            p_split (string): Data split used.
        """

        # Super class init.
        super(DFaustDS, self).__init__()

        # Save parameters.
        self.path_ = Path(p_data_folder)

        if p_split == "train":
            self.path_ = self.path_ / "train"
        else:
            self.path_ = self.path_ / "test"

        self.files_ = [
            f for f in self.path_.iterdir() if f.is_file() and f.suffix == ".pt"
        ]
        self.num_pts_ = p_num_pts

        # self.class_names_ = [
        #        'butt', 'left_thigh', 'right_thigh', 'mid_belly', 'left_calf', 'right_calf',
        #        'upper_belly', 'right_foot', 'left_foot', 'upper_thorax', 'nothing1', 'nothing2',
        #        'neck', 'right_shoulder', 'left_shoulder', 'head', 'right_upper_arm', 'left_upper_arm',
        #        'right_forearm', 'left_forearm', 'right_hand', 'left_hand', 'nothing3']
        self.class_names_ = [
            "butt",
            "left_thigh",
            "right_thigh",
            "mid_belly",
            "left_calf",
            "right_calf",
            "upper_belly",
            "right_foot",
            "left_foot",
            "upper_thorax",
            "neck",
            "right_shoulder",
            "left_shoulder",
            "head",
            "right_upper_arm",
            "left_upper_arm",
            "right_forearm",
            "left_forearm",
            "right_hand",
            "left_hand",
        ]

        # Configure the data augmentation pipeline.
        if len(p_augmentation_cfg) > 0:
            self.aug_pipeline_ = pclib.augment.AugPipeline()
            self.aug_pipeline_.create_pipeline(p_augmentation_cfg)
        else:
            self.aug_pipeline_ = None

    def __len__(self):
        """Get the lenght of the data set.

        Returns:
            (int) Number of models.
        """
        return int(
            len(self.files_) / 2
        )  # since always two files per model (labels and points)

    def increase_epoch_counter(self):
        """Method to increase the epoch counter for user-defined augmentations."""
        if not self.aug_pipeline_ is None:
            self.aug_pipeline_.increase_epoch_counter()

    def __getitem__(self, idx):
        """Get item in position idx.

        Args:
            idx (int): Index of the element to return.

        Returns:
            (torch.Tensors) Point clouds and labels.
        """
        model_path = self.path_ / f"model_{idx}_pc.pt"
        labels_path = self.path_ / f"model_{idx}_labels.pt"

        pts = torch.load(model_path, map_location="cpu").to(torch.float32)[
            : self.num_pts_, :
        ]
        labels = torch.load(labels_path, map_location="cpu").to(torch.float32)[
            : self.num_pts_
        ]
        # remove 10,11, 22 labels -
        mask = torch.where(labels > 9)
        labels[mask] -= 2
        features = torch.ones((pts.shape[0], 1)).to(torch.float32)

        if self.aug_pipeline_:
            pts, _, _ = self.aug_pipeline_.augment(pts)

        return pts, labels, features, idx
