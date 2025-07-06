import numpy as np
import torch
import point_cloud_lib_ops

from torch_scatter import scatter_max, scatter_min

class BallQuery(torch.autograd.Function):
    """Function to compute a ball query.
    """

    @staticmethod
    def forward(
        p_ctx, 
        p_pt_src,
        p_pt_sample,
        p_batch_id_src,
        p_batch_id_sample,
        radius,
        max_neighbors):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pt_src (tensor nxk): Source point cloud. 
            p_pt_sample (tensor mxk): Sample point cloud. 
            p_batch_id_src (tensor n): Source batch ids.
            p_batch_id_sample (tensor m): Source batch ids.
            radius (float): Radius of the ball query.
            max_neighbors (int): Maximum number of neighbors.
        Returns:
            tensor lx2: Tensor with the neighbor information.
            tensor m: Tensor with the end neighbor position.
        """
        
        # Compute grid information.
        min_pt = scatter_min(p_pt_src, p_batch_id_src.to(torch.int64), dim=0)[0] - 1e-6
        max_pt = scatter_max(p_pt_src, p_batch_id_src.to(torch.int64), dim=0)[0] - 1e-6
        num_cells = (max_pt - min_pt)/radius
        num_cells = torch.max(num_cells.to(torch.int32) + 1, dim=0)[0]
        radius_tensor = torch.from_numpy(np.array(
            [radius for _ in range(p_pt_src.shape[1])])).to(torch.float32).to(p_pt_src.device)

        # Execute operation.
        neighbors, start_ids = point_cloud_lib_ops.ball_query(
            p_pt_src.to(torch.float32),
            p_pt_sample.to(torch.float32),
            p_batch_id_src.to(torch.int32),
            p_batch_id_sample.to(torch.int32),
            min_pt.to(torch.float32),
            num_cells.to(torch.int32),
            radius_tensor,
            max_neighbors)
        
        return neighbors, start_ids


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """
        

        return None, None, None, None, None, None