import torch

from point_cloud_lib.pc import Pointcloud

class DropPathPC(torch.nn.Module):
    """Class to implement a drop path for point clouds.
    """

    def __init__(self, p_drop_prob):
        """Constructor.

        Args:
            p_drop_prob (float): Drop probability
        """

        # Save parameters.
        self.drop_prob_ = p_drop_prob

        # Super constructor.
        torch.nn.Module.__init__(self)


    def forward(self, p_x, p_pc):
        """Forward method.

        Args:
            p_x (tensor): Input feature tensor.
            p_pc (Pointcloud): Input point cloud.

        Return:
            out_x (tensor): Output tensor.
        """

        if self.drop_prob_ == 0. or not self.training:
            return p_x
        
        keep_prob = 1 - self.drop_prob_
        random_tensor = keep_prob + \
            torch.rand((p_pc.batch_size_,), 
                        dtype=p_x.dtype, device=p_x.device)
        random_tensor.floor_()  # binarize
        if hasattr(p_pc, 'batch_ids_considering_frames_'):
            random_tensor = torch.index_select(
                random_tensor, 0, p_pc.batch_ids_considering_frames_.to(torch.int64))
        else:
            random_tensor = torch.index_select(
                random_tensor, 0, p_pc.batch_ids_.to(torch.int64))
        
        output = p_x.div(keep_prob) * random_tensor.reshape((-1,1))
        return output