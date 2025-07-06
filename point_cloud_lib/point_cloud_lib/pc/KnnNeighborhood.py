from abc import ABC, abstractmethod
import numpy as np
import torch

from torch_scatter import scatter_add

from torch_cluster import knn, knn_graph

from point_cloud_lib.pc import Neighborhood
from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.custom_ops import KNNQuery


class KnnNeighborhood(Neighborhood):
    """Class to represent a k-nn neighborhood."""

    def __init__(
        self, p_pc_src, p_samples, p_k, p_keep_empty=False, p_standard_knn=False
    ):
        """Constructor.

        Args:
            p_pc_src (Pointcloud): Source point cloud.
            p_pc_samples (Pointcloud): Sample point cloud.
            p_radius (float): Radius.
            p_keep_empty (bool): Boolean that indicates if we keep empty neighboors.
        """

        # Store variables.
        self.k_ = p_k
        self.keep_empty_ = p_keep_empty
        self.standard_knn_ = p_standard_knn

        # Super class init.
        super(KnnNeighborhood, self).__init__(p_pc_src, p_samples)

    def __compute_neighborhood__(self):
        """Abstract mehod to implement the neighborhood selection."""
        if self.pc_src_ == self.samples_ and self.k_ <= 64 and not self.standard_knn_:

            cur_neighs = KNNQuery.apply(
                self.pc_src_.pts_, self.pc_src_.batch_ids_, self.k_
            )
            center_indices = torch.arange(
                start=0,
                end=self.pc_src_.pts_.shape[0],
                dtype=torch.int32,
                device=self.pc_src_.pts_.device,
            )
            center_indices = center_indices.unsqueeze(1).repeat(1, self.k_)
            self.neighbors_ = torch.cat(
                (center_indices.reshape((-1, 1)), cur_neighs.reshape((-1, 1))), -1
            )

            if self.keep_empty_:
                self.start_ids_ = (
                    torch.arange(
                        start=0,
                        end=self.pc_src_.pts_.shape[0],
                        dtype=torch.int32,
                        device=self.pc_src_.pts_.device,
                    )
                    * self.k_
                    + self.k_
                )
            else:
                mask = self.neighbors_[:, 1] >= 0
                self.neighbors_ = self.neighbors_[mask]
                #
                self.start_ids_ = scatter_add(
                    torch.ones_like(self.neighbors_[:, 1], dtype=torch.int32),
                    self.neighbors_[:, 0].to(torch.int64),
                    dim=0,
                )
                self.start_ids_ = torch.cumsum(self.start_ids_, 0)
        #
        else:
            self.neighbors_ = knn(
                x=self.pc_src_.pts_,
                y=self.samples_.pts_,
                k=self.k_,
                batch_x=self.pc_src_.batch_ids_,
                batch_y=self.samples_.batch_ids_,
            )
            self.neighbors_ = torch.transpose(self.neighbors_, 0, 1).contiguous()

            self.start_ids_ = scatter_add(
                torch.ones_like(self.neighbors_[:, 1], dtype=torch.int32),
                self.neighbors_[:, 0],
                dim=0,
            )

            if self.keep_empty_:
                padding = self.k_ - self.start_ids_
                new_padding = torch.cumsum(padding, 0) - padding
                new_neighbors = (
                    torch.ones(
                        (self.start_ids_.shape[0], self.k_, 2),
                        dtype=self.neighbors_.dtype,
                        device=self.start_ids_.device,
                    )
                    * -1
                )
                new_neighbors[:, :, 0] = torch.arange(
                    start=0,
                    end=self.start_ids_.shape[0],
                    dtype=self.neighbors_.dtype,
                    device=self.start_ids_.device,
                ).reshape((-1, 1))

                new_indices = (
                    torch.arange(
                        start=0,
                        end=self.neighbors_.shape[0],
                        dtype=torch.int32,
                        device=self.start_ids_.device,
                    )
                    + new_padding[self.neighbors_[:, 0]]
                )
                new_neighbors = new_neighbors.reshape((-1, 2))
                new_neighbors[new_indices, :] = self.neighbors_

                self.start_ids_ = (
                    torch.arange(
                        start=0,
                        end=self.start_ids_.shape[0],
                        dtype=torch.int32,
                        device=self.start_ids_.device,
                    )
                    * self.k_
                    + self.k_
                )
                self.neighbors_ = new_neighbors
            else:
                self.start_ids_ = torch.cumsum(self.start_ids_, 0)
