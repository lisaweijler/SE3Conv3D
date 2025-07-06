import numpy as np
import torch
from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_add

class Pointcloud(object):
    """Class to represent a point cloud.
    """

    def __init__(self, p_pts, p_batch_ids, **kwargs):
        '''Constructor.
        
        Args:
            p_pts (np.array nxd): Point coordinates.
            p_batch_ids (np.array n): Point batch ids.
        '''

        # Check if requires_grad is specified.
        self.pts_with_grads_ = False
        if 'requires_grad' in kwargs:
            self.pts_with_grads_ = kwargs['requires_grad']
            del kwargs['requires_grad']

        # Create the tensors.
        self.pts_ = torch.as_tensor(p_pts, **kwargs)
        self.batch_ids_ = torch.as_tensor(p_batch_ids, **kwargs)
        self.batch_size_ = torch.max(self.batch_ids_) + 1

        # Update requires_grad if needed.
        if self.pts_with_grads_:
            self.pts_.requires_grad = True


    def to_device(self, p_device):
        """Method to move the tensors to a specific device.

        Return:
            device p_device: Destination device.
        """
        self.pts_ = self.pts_.to(p_device)
        self.batch_ids_ = self.batch_ids_.to(p_device)
        self.batch_size_ = self.batch_size_.to(p_device)

    
    def get_num_points_per_batch(self):
        """Method to get the number of points per batch.

        Return:
            int tensor: Number of points per each batch.
        """
        with torch.no_grad():
            aux_ones = torch.ones_like(self.batch_ids_)
            num_pts = torch.zeros((self.batch_size_))\
                .to(torch.int32).to(self.batch_ids_.device)
            num_pts.index_add_(0, self.batch_ids_.to(torch.int64), aux_ones)
        return num_pts


    def global_pooling(self, p_in_tensor, p_pooling_method = "avg"):
        """Method to perform a global pooling over a set of features.

        Args:
            p_in_tensor (tensor pxd): Tensor to pool.
            p_pooling_method (string): Pooling method (avg, max, min)

        Return:
            tensor bxd: Pooled tensor.
        """
        batch_id_indexs = self.batch_ids_.to(torch.int64)
        if p_pooling_method == "max":
            return scatter_max(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "min":
            return scatter_min(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "avg":
            return scatter_mean(p_in_tensor, batch_id_indexs, dim=0)
        elif p_pooling_method == "sum":
            return scatter_add(p_in_tensor, batch_id_indexs, dim=0)


    def global_upsample(self, p_in_tensor):
        """Method to perform a global upsample over a set of features.

        Args:
            p_in_tensor (tensor bxd): Tensor to upsample.

        Return:
            tensor pxd: Upsampled tensor.
        """
        return torch.index_select(p_in_tensor, 0, self.batch_ids_.to(torch.int64))


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Points:\n{}\n"\
                "### Batch Ids:\n{}\n"\
                "### Batch Size:\n{}\n"\
                "### Pdf:\n{}"\
                .format(self.pts_, self.batch_ids_, 
                self.batch_size_, self.pts_pdf_)


    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Method to apply torch functions to the object.
        """
        if kwargs is None:
            kwargs = {}
        args = [a.pts_ if hasattr(a, 'pts_') else a for a in args]
        ret = func(*args, **kwargs)
        return Pointcloud(ret, self.batch_ids_)

