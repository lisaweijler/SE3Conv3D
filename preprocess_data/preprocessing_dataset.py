"""This file contains the definition of different heterogeneous datasets used for training."""

import joblib
import numpy as np
import torch
from pathlib import Path
import tqdm.auto as tqdm
import webdataset as wds


class AMASSPreLoadDataset(torch.utils.data.Dataset):
    def __init__(self, src_data_path, is_train):
        # logger.info(f"Loading AMASS dataset: --> training: {is_train}")
        print(f"## Loading AMASS dataset: --> training: {is_train}")

        self.source_data_path = src_data_path
        self.is_train = is_train
        self.marker_data = []
        self.rotations = []
        self.translation = []
        self.fnames = []
        self.betas = []

        self.marker_data, self.rotations, self.translation, self.betas, self.fnames = (
            self.unpack_data()
        )

        if self.is_train:
            print(
                f"## Finished loading all the train datasets. Total number of samples: {len(self.fnames)}"
            )
        else:
            print(
                f"## Finished loading all the val datasets. Total number of samples: {len(self.fnames)}"
            )

    def unpack_data(self):
        marker_data = []
        rotations = []
        translation = []
        fnames = []
        betas = []
        if self.is_train:
            data_path = Path(self.source_data_path) / "DFaust_67_train.pth.tar"

            data = joblib.load(str(data_path))

            print(f"Loaded dataset: ----> {data_path} ")
            print(f"Number of Sequence: ----> {len(data)} ")

            for i, x in tqdm.tqdm(enumerate(data)):
                fnames.append(data[i]["fname"])
                marker_data.append(torch.from_numpy(data[i]["markers"]))  # numpy
                seq_length = data[i]["markers"].shape[0]
                rotations.append(torch.from_numpy(data[i]["poses"]))
                translation.append(torch.from_numpy(data[i]["trans"]))

                betas.append(
                    torch.from_numpy(
                        np.repeat(data[i]["betas"][np.newaxis, :], seq_length, axis=0)
                    )
                )

            marker_data = torch.cat(marker_data)
            rotations = torch.cat(rotations)
            translation = torch.cat(translation)
            betas = torch.cat(betas)

        else:
            data_path = Path(self.source_data_path) / "MPI_Limits"
            for tar_path in tqdm.tqdm(
                sorted(data_path.glob("*.tar"), key=lambda x: x.stem)
            ):
                dataset = wds.WebDataset(str(tar_path)).decode().to_tuple("input.pth")
                for i, (batch,) in enumerate(dataset):
                    fnames.append(batch["fname"])
                    marker_data.append(batch["markers"])

                    rotations.append(batch["poses"])

                    translation.append(batch["trans"])  # torch

                    betas.append(batch["betas"])

            marker_data = torch.stack(marker_data)
            rotations = torch.stack(rotations)
            translation = torch.stack(translation)
            betas = torch.stack(betas)

        return marker_data, rotations, translation, betas, fnames

    def __len__(self):
        return self.translation.shape[0]

    def __getitem__(self, index):
        item = dict()

        motion_rotations = self.rotations[index].reshape(-1, 3)
        translation = self.translation[index]
        body_shape = self.betas[index]

        item["rotations"] = motion_rotations.type(dtype=torch.float32)
        item["translation"] = translation.type(dtype=torch.float32)
        item["body_shape"] = body_shape.type(dtype=torch.float32)

        return item
