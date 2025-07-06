import numpy as np
import torch
import point_cloud_lib_ops

from torch_scatter import scatter_max, scatter_min

class ComputeKeys(torch.autograd.Function):
    """Function to compute the keys of a voxelization.
    """

    @staticmethod
    def forward(
        p_ctx, 
        p_pts,
        p_batch_ids,
        p_aabb_min,
        p_grid_size,
        p_cell_size):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxk): Point cloud. 
            p_batch_ids (tensor n): Batch ids.
            p_aabb_min (tensor bxk): Minimum point bounding box.
            p_grid_size (tensor k): Grid size.
            p_cell_size (tensor k): Cell size.
        Returns:
            tensor lx2: Tensor with the neighbor information.
            tensor m: Tensor with the end neighbor position.
        """
        

        # Execute operation.
        return point_cloud_lib_ops.compute_keys(
            p_pts.to(torch.float32),
            p_batch_ids.to(torch.int32),
            p_aabb_min.to(torch.float32),
            p_grid_size.to(torch.int32),
            p_cell_size.to(torch.float32))



    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """
        

        return None, None, None, None, None