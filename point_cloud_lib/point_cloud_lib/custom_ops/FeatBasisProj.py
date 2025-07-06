import torch
import point_cloud_lib_ops

class FeatBasisProj(torch.autograd.Function):
    """Function to compute the feature projection on a set of basis.
    """

    @staticmethod
    #@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        p_ctx, 
        p_pt_basis,
        p_pt_features,
        p_neighbors,
        p_start_ids):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pt_basis (tensor mxk): Neighbor embedding basis functions. 
            p_pt_features (tensor nxf): Point features.
            p_neighbors (tensor mx2): Neighbor list.
            p_start_ids (tensor n): Start indices of each point.
        Returns:
            tensor nxfxk: Tensor of point features and basis functions.
        """

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_pt_basis,
            p_pt_features,
            p_neighbors,
            p_start_ids)
        
        # Execute operation.
        result_tensor = point_cloud_lib_ops.feat_basis_proj(
            p_pt_basis.to(torch.float32),
            p_pt_features.to(torch.float32),
            p_neighbors.to(torch.int32),
            p_start_ids.to(torch.int32))
        
        return result_tensor


    @staticmethod
    #@torch.cuda.amp.custom_bwd
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """
        
        # get saved tensors.
        pt_basis, pt_features, neighbors, start_ids = p_ctx.saved_tensors

        # Compute the gradients.
        feat_grads, basis_grads = point_cloud_lib_ops.feat_basis_proj_grad(
            pt_basis.to(torch.float32), 
            pt_features.to(torch.float32), 
            neighbors.to(torch.int32), 
            start_ids.to(torch.int32), 
            p_grads)

        return basis_grads.to(pt_basis.dtype), feat_grads.to(pt_features.dtype), None, None