import torch
import point_cloud_lib_ops

from torch_scatter import scatter_add

class TransformNeighConv(torch.autograd.Function):

    @staticmethod
    def forward(
        p_ctx, 
        p_score,
        p_mats,
        p_feats,
        p_neighbors):

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_score,
            p_mats,
            p_feats,
            p_neighbors)
        
        # Compute the kernel.
        cur_kernel = torch.einsum('nk,iko->nio', p_score, p_mats)
        
        # Actual convolution.
        cur_feats = p_feats[p_neighbors[:,1],:]
        conv_results = torch.einsum('nio,ni->no', cur_kernel, cur_feats)
        
        return conv_results


    @staticmethod
    def backward(p_ctx, p_grads):
        
        # get saved tensors.
        p_score, p_mats, p_feats, p_neighbors = p_ctx.saved_tensors

        # Compute the kernel.
        cur_kernel = torch.einsum('nk,iko->nio', p_score, p_mats)   

        feat_grads = torch.einsum('nio,no->ni', cur_kernel, p_grads)
        feat_grads = scatter_add(feat_grads, p_neighbors[:,1], dim=0)

        cur_feats = p_feats[p_neighbors[:,1],:]
        kernel_grads = torch.einsum('no,ni->nio', p_grads, cur_feats)
        score_grads = torch.einsum('nio,iko->nk', kernel_grads, p_mats) 
        mat_grads = torch.einsum('nk,nio->iko', p_score, kernel_grads)   

        return score_grads, mat_grads, feat_grads, None