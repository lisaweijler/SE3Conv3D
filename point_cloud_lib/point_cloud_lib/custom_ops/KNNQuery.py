import numpy as np
import torch
import point_cloud_lib_ops

from torch_scatter import scatter_max, scatter_min

class KNNQuery(torch.autograd.Function):
    """Function to compute a knn query.
    """

    @staticmethod
    def forward(
        p_ctx, 
        p_pt_src,
        p_batch_id_src,
        p_k):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pt_src (tensor nxk): Source point cloud. 
            p_batch_id_src (tensor n): Source batch ids.
            p_k (int): Maximum number of neighbors.
        Returns:
            tensor lx2: Tensor with the neighbor information.
            tensor m: Tensor with the end neighbor position.
        """
        
        # Execute operation.
        neighbors = point_cloud_lib_ops.knn_query(
            p_pt_src.to(torch.float32),
            p_batch_id_src.to(torch.int32),
            p_k)
        
        return neighbors


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """
        

        return None, None, None, None, None, None