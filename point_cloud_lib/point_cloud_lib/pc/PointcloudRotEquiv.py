from einops import rearrange, repeat
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_add
from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.pc import sample_reference_frames
from point_cloud_lib.pc import sample_reference_frames_pca
from point_cloud_lib.pc import sample_global_reference_frames_pca
from point_cloud_lib.pc import KnnNeighborhood
from point_cloud_lib.pc import BQNeighborhood


class PointcloudRotEquiv(Pointcloud):
    """Class to represent a point cloud."""

    def __init__(
        self,
        p_pts,
        p_batch_ids,
        p_ref_frames_config,
        ref_frames_pts=None,
        standard_knn=False,
        **kwargs
    ):
        """Constructor.

        Args:
            p_pts (np.array nxd): Point coordinates.
            p_ref_frames_config (dict): configurations for frame sampling.
            p_batch_ids (np.array n): Point batch ids.
        """
        super(PointcloudRotEquiv, self).__init__(p_pts, p_batch_ids, **kwargs)
        # Init neighborhood cache.
        self.neigh_cache_ = {}
        self.local_frames_pca_cache_ = {}

        self.local_frames_config_ = p_ref_frames_config
        self.standard_knn_ = standard_knn

        self.ref_frames_pts = ref_frames_pts
        p_local_frames = self.get_local_ref_frames()
        self.n_frames_ = p_local_frames.shape[1]

        self.local_frames_ = torch.as_tensor(p_local_frames, **kwargs)

        self.batch_ids_considering_frames_ = repeat(
            self.batch_ids_, "n -> (n times)", times=self.n_frames_
        )

        # Update requires_grad if needed.
        if self.pts_with_grads_:
            self.local_frames_.requires_grad = True

    def get_ref_frame_neighborhood(self, p_neigh_method, **kwargs):
        neigh_str = str(p_neigh_method)
        if p_neigh_method == "knn":
            neigh_str = neigh_str + str(kwargs["neigh_k"])
        elif p_neigh_method == "ball_query":
            neigh_str = neigh_str + str(kwargs["bq_radius"])
        if neigh_str in self.neigh_cache_:
            return self.neigh_cache_[neigh_str]
        else:
            if p_neigh_method == "knn":
                neighborhood = KnnNeighborhood(
                    self,
                    self,
                    kwargs["neigh_k"],
                    p_keep_empty=True,
                    p_standard_knn=self.standard_knn_,
                )  # to make sure to have same amount of neighbors even if pc is too small
            elif p_neigh_method == "ball_query":
                neighborhood = BQNeighborhood(self, self, kwargs["bq_radius"])

            self.neigh_cache_[neigh_str] = neighborhood
            return neighborhood

    def get_local_ref_frames(self):

        # get global ref points
        if self.ref_frames_pts is not None:
            batch_size = self.pts_.shape[0]  # one point per batch
            if self.local_frames_config_["pca"]:
                # compute neighborhood
                if "se3-all" not in self.local_frames_pca_cache_:

                    self.local_frames_pca_cache_["se3-all"] = (
                        sample_global_reference_frames_pca(
                            rearrange(
                                self.ref_frames_pts, "(b n) f -> b n f", b=batch_size
                            ),
                            axis_fixed=self.local_frames_config_["fixed_axis"],
                            device=self.pts_.device,
                        )
                    )  # batchsize x 4 x 9

                # shuffle frames

                n_points = self.local_frames_pca_cache_["se3-all"].shape[0]  # batchsize
                n_frames = self.local_frames_pca_cache_["se3-all"].shape[1]  # 4
                weights = torch.ones(n_frames, device=self.pts_.device).expand(
                    n_points, -1
                )
                indices_permutations = torch.multinomial(
                    weights, num_samples=n_frames, replacement=False
                )

                shuffled_frames = torch.gather(
                    self.local_frames_pca_cache_["se3-all"],
                    1,
                    indices_permutations[:, :, None].expand(
                        -1, -1, self.local_frames_pca_cache_["se3-all"].shape[-1]
                    ),
                )
                # select number of frames specified
                ref_frames = shuffled_frames[
                    :, : self.local_frames_config_["n_frames"], :
                ]

            else:
                ref_frames = sample_reference_frames(
                    n_origins=1,
                    n_frames=self.local_frames_config_["n_frames"],
                    axis_fixed=self.local_frames_config_["fixed_axis"],
                    device=self.pts_.device,
                )

        else:
            if self.local_frames_config_["pca"]:
                # compute neighborhood
                if "se3-all" not in self.local_frames_pca_cache_:

                    neighborhood = self.get_ref_frame_neighborhood(
                        self.local_frames_config_["neigh_method"],
                        **self.local_frames_config_["neigh_kwargs"]
                    )

                    self.local_frames_pca_cache_["se3-all"] = (
                        sample_reference_frames_pca(
                            self.pts_,
                            neighborhood,
                            axis_fixed=self.local_frames_config_["fixed_axis"],
                            device=self.pts_.device,
                        )
                    )  # n x 4 x 9

                # shuffle frames

                n_points = self.local_frames_pca_cache_["se3-all"].shape[0]
                n_frames = self.local_frames_pca_cache_["se3-all"].shape[1]  # 4
                weights = torch.ones(n_frames, device=self.pts_.device).expand(
                    n_points, -1
                )
                indices_permutations = torch.multinomial(
                    weights, num_samples=n_frames, replacement=False
                )

                shuffled_frames = torch.gather(
                    self.local_frames_pca_cache_["se3-all"],
                    1,
                    indices_permutations[:, :, None].expand(
                        -1, -1, self.local_frames_pca_cache_["se3-all"].shape[-1]
                    ),
                )
                # select number of frames specified

                ref_frames = shuffled_frames[
                    :, : self.local_frames_config_["n_frames"], :
                ]

            else:
                ref_frames = sample_reference_frames(
                    n_origins=self.pts_.shape[0],
                    n_frames=self.local_frames_config_["n_frames"],
                    axis_fixed=self.local_frames_config_["fixed_axis"],
                    device=self.pts_.device,
                )

        return ref_frames

    def to_device(self, p_device):
        """Overload from baseclass
        Method to move the tensors to a specific device.

        Return:
            device p_device: Destination device.
        """
        self.pts_ = self.pts_.to(p_device)
        self.local_frames_ = self.local_frames_.to(p_device)
        self.batch_ids_ = self.batch_ids_.to(p_device)
        self.batch_size_ = self.batch_size_.to(p_device)
        self.batch_ids_considering_frames_ = self.batch_ids_considering_frames_.to(
            p_device
        )

    def global_pooling_specific_feature_pooling(
        self, p_in_tensor, p_global_pooling_method="avg", p_feature_pooling_method="avg"
    ):
        """
        Method to perform a pooling over a set of features per point and then global pooling.
        E.g. if we have 4 frames -> 4 features, we need one at the end of
        segementation task and another global pooling for classification.
        Here the pooling methods for features and global can be different.

        Args:
            p_in_tensor (tensor pxd): Tensor to pool.
            p_pooling_method (string): Pooling method (avg, max, min)

        Return:
            tensor bxd: Pooled tensor.
        """
        pooled_features_x = self.feature_pooling(
            p_in_tensor, p_pooling_method=p_feature_pooling_method
        )
        batch_id_indexs = self.batch_ids_.to(torch.int64)
        if p_global_pooling_method == "max":
            return scatter_max(pooled_features_x, batch_id_indexs, dim=0)[0]
        elif p_global_pooling_method == "min":
            return scatter_min(pooled_features_x, batch_id_indexs, dim=0)[0]
        elif p_global_pooling_method == "avg":
            return scatter_mean(pooled_features_x, batch_id_indexs, dim=0)
        elif p_global_pooling_method == "sum":
            return scatter_add(pooled_features_x, batch_id_indexs, dim=0)

    def feature_pooling(self, p_in_tensor, p_pooling_method="avg"):
        """
        Method to perform a pooling over a set of features per point.
        E.g. if we have 4 frames -> 4 features, we need one at the end of
        segementation task.

        Args:
            p_in_tensor (tensor pxd): Tensor to pool.
            p_pooling_method (string): Pooling method (avg, max, min)

        Return:
            tensor bxd: Pooled tensor.
        """
        range_features = torch.tensor(
            range(self.pts_.shape[0]), device=self.pts_.device, dtype=torch.int64
        )

        batch_id_indexs = repeat(
            range_features, "n -> (n t)", t=self.local_frames_config_["n_frames"]
        )
        if p_pooling_method == "max":
            return scatter_max(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "min":
            return scatter_min(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "avg":
            return scatter_mean(p_in_tensor, batch_id_indexs, dim=0)
        elif p_pooling_method == "sum":
            return scatter_add(p_in_tensor, batch_id_indexs, dim=0)

    def global_pooling(self, p_in_tensor, p_pooling_method="avg"):
        """Overload from baseclass
        Method to perform a global pooling over a set of features.

        Args:
            p_in_tensor (tensor pxd): Tensor to pool.
            p_pooling_method (string): Pooling method (avg, max, min)

        Return:
            tensor bxd: Pooled tensor.
        """
        batch_id_indexs = self.batch_ids_considering_frames_.to(torch.int64)
        if p_pooling_method == "max":
            return scatter_max(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "min":
            return scatter_min(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "avg":
            return scatter_mean(p_in_tensor, batch_id_indexs, dim=0)
        elif p_pooling_method == "sum":
            return scatter_add(p_in_tensor, batch_id_indexs, dim=0)

    def global_upsample(self, p_in_tensor):
        """Overload from baseclass
        Method to perform a global upsample over a set of features.

        Args:
            p_in_tensor (tensor bxd): Tensor to upsample.

        Return:
            tensor pxd: Upsampled tensor.
        """
        return torch.index_select(
            p_in_tensor, 0, self.batch_ids_considering_frames_.to(torch.int64)
        )

    def __repr__(self):
        """Overload from baseclass
        Method to create a string representation of
            object.
        """
        return (
            "### Points:\n{}\n"
            "### Local Ref Frames:\n{}\n"
            "### Batch Ids:\n{}\n"
            "### Batch Size:\n{}\n"
            "### Pdf:\n{}".format(
                self.pts_,
                self.local_frames_,
                self.batch_ids_,
                self.batch_size_,
                self.pts_pdf_,
            )
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Overload from baseclass
        Method to apply torch functions to the object.
        """
        if kwargs is None:
            kwargs = {}
        args = [a.pts_ if hasattr(a, "pts_") else a for a in args]
        ret = func(*args, **kwargs)
        return PointcloudRotEquiv(ret, self.batch_ids_, self.local_frames_)
