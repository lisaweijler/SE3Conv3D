import torch

class LinearPNE(torch.autograd.Function):
    """Function to compute a linear point neighborhood embeddings.
    """

    @staticmethod
    def forward(
        p_ctx, 
        p_pts,
        p_samples,
        p_neighbors,
        p_proj_axes,
        p_proj_biases,
        p_norm_dist):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordinates.
            p_samples (tensor n'xd): Point sample coordinates.
            p_neighbors (tensor mx2): Neighbor list.
            p_proj_axes (tensor dxk): Projection axes.
            p_proj_biases (tensor k): Projection biases.
            p_norm_dist (float): Normalization distance.
        Returns:
            tensor mxk: Tensor of point neighborhood embeddings.
        """

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_pts,
            p_samples,
            p_neighbors,
            p_norm_dist)
        
        # Execute operation.
        rel_pt = (p_pts[p_neighbors[:,1],:] - p_samples[p_neighbors[:,0],:])*p_norm_dist
        pt_basis = torch.matmul(rel_pt, p_proj_axes) + p_proj_biases.reshape((1, -1))
        
        return pt_basis


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxk): Input gradients.
        """
        
        # get saved tensors.
        pts, samples, neighbors, norm_dist = p_ctx.saved_tensors

        # Compute the gradients.
        rel_pt = (pts[neighbors[:,1],:] - samples[neighbors[:,0],:])*norm_dist
        proj_axes_grad = torch.matmul(rel_pt.T, p_grads) 
        proj_biases_grad = torch.sum(p_grads, 0)

        return None, None, None, proj_axes_grad, proj_biases_grad, None
    

class KPPNE(torch.autograd.Function):
    """Function to compute a kernel point neighborhood embeddings.
    """

    @staticmethod
    def forward(
        p_ctx, 
        p_pts,
        p_samples,
        p_neighbors,
        p_kpts,
        p_sigma,
        p_proj_axes,
        p_proj_biases,
        p_norm_dist,
        p_corr_func):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordinates.
            p_samples (tensor n'xd): Point sample coordinates.
            p_neighbors (tensor mx2): Neighbor list.
            p_kpts (tensor k1xd): Kernel points.
            p_sigma (float): Sigma value for the kernel points
                relative position.
            p_proj_axes (tensor k1xk2): Projection axes.
            p_proj_biases (tensor k2): Projection biases.
            p_norm_dist (float): Normalization distance.
            p_corr_func (string): Correlation function.
        Returns:
            tensor mxk: Tensor of point neighborhood embeddings.
        """

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_pts,
            p_samples,
            p_neighbors,
            p_kpts,
            p_norm_dist)
        p_ctx.sigma_ = p_sigma
        p_ctx.corr_funct_ = p_corr_func
        
        # Kernel points relative positions.
        rel_pt = (p_pts[p_neighbors[:,1],:] - p_samples[p_neighbors[:,0],:])*p_norm_dist
        rel_pt = rel_pt.reshape((-1, 1, p_pts.shape[-1])) - p_kpts.reshape((1, -1, p_pts.shape[-1]))
        distances = torch.sqrt(torch.sum(rel_pt**2, -1))
        distances = distances/p_sigma

        # Correlation function.
        if p_corr_func == "gauss":
            distances = torch.exp(-(distances**2)/2.)
        elif p_corr_func == "linear":
            distances = torch.clamp(1.0 - distances, min=0.0)
        elif p_corr_func == "box":
            min_dist_id = torch.argmin(distances, -1)
            distances = torch.nn.functional.one_hot(
                min_dist_id, num_classes=p_kpts.shape[0]).to(torch.float32)

        # Final projection
        pt_basis = torch.matmul(distances, p_proj_axes) + p_proj_biases.reshape((1, -1))
        
        return pt_basis


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxk): Input gradients.
        """
        
        # get saved tensors.
        pts, samples, neighbors, kpts, norm_dist = p_ctx.saved_tensors
        sigma = p_ctx.sigma_
        corr_funct = p_ctx.corr_funct_

        # Kernel points relative positions.
        rel_pt = (pts[neighbors[:,1],:] - samples[neighbors[:,0],:])*norm_dist
        rel_pt = rel_pt.reshape((-1, 1, pts.shape[-1])) - kpts.reshape((1, -1, pts.shape[-1]))
        distances = torch.sqrt(torch.sum(rel_pt**2, -1))
        distances = distances/sigma

        # Correlation function.
        if corr_funct == "gauss":
            distances = torch.exp(-(distances**2)/2.)
        elif corr_funct == "linear":
            distances = torch.clamp(1.0 - distances, min=0.0)
        elif corr_funct == "box":
            min_dist_id = torch.argmin(distances, -1)
            distances = torch.nn.functional.one_hot(
                min_dist_id, num_classes=kpts.shape[0]).to(torch.float32)

        # Compute the gradients.
        proj_axes_grad = torch.matmul(distances.T, p_grads) 
        proj_biases_grad = torch.sum(p_grads, 0)

        return None, None, None, None, None, proj_axes_grad, proj_biases_grad, None, None