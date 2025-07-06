from abc import ABC, abstractmethod
import math
import numpy as np
import torch

from torch_scatter import scatter_add

from point_cloud_lib.layers import IConvLayer, IConvLayerFactory, create_pts_icosphere
from point_cloud_lib.custom_ops import FeatBasisProj, KPPNE

class MultiHeadAttLayer(IConvLayer):
    """Low-Rank attention point convolution.
    """

    def __init__(self,
        p_dims,
        p_in_features, 
        p_out_features,
        p_num_basis,
        p_kp_res,
        p_num_heads):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_num_basis (int): Number of basis.
            p_kp_res (string): Kernel points resolution.
            p_num_heads (int): Number of heads.
        """

        # Super class init.
        super(MultiHeadAttLayer, self).__init__(
            p_dims, p_in_features, p_out_features)

        assert(p_dims == 3)
        
        # Save parameters.
        self.num_basis_ = p_num_basis
        self.kp_res_ = p_kp_res
        self.num_heads_ = p_num_heads
        self.value_size_ = p_in_features

        # Create the kernel points.
        if self.kp_res_ == "single":        
            self.kp_sigma_ = 0.3
            kp_scale = 0.6
            kp = create_pts_icosphere(0)
            kp = np.concatenate((kp, np.array([[0.0,0.0,0.0]])))
            kp = kp.astype(np.float32)*kp_scale
        elif self.kp_res_ == "double":
            self.kp_sigma_ = 0.16
            kp_scale = 0.35
            kp = create_pts_icosphere(0)*kp_scale
            kp_2 = create_pts_icosphere(1)*kp_scale*2
            kp = np.concatenate((kp, kp_2))
            kp = np.concatenate((kp, np.array([[0.0,0.0,0.0]])))
            kp = kp.astype(np.float32)
        
        # Random rotate kernel points.
        cur_angle = np.random.uniform(size=(3,))*2.*np.pi
        Rx = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(cur_angle[0]), -np.sin(cur_angle[0])],
                        [0.0, np.sin(cur_angle[0]), np.cos(cur_angle[0])]]).astype(np.float32)
        Ry = np.array([[np.cos(cur_angle[1]), 0.0, np.sin(cur_angle[1])],
                        [0.0, 1.0, 0.0],
                        [-np.sin(cur_angle[1]), 0.0, np.cos(cur_angle[1])]]).astype(np.float32)
        Rz = np.array([[np.cos(cur_angle[2]), -np.sin(cur_angle[2]), 0.0],
                        [np.sin(cur_angle[2]), np.cos(cur_angle[2]), 0.0],
                        [0.0, 0.0, 1.0]]).astype(np.float32)
        R = np.dot(np.dot(Rx, Ry), Rz)
        kp = np.dot(kp, R)

        # Save kernel points.
        self.register_buffer('kernel_pts_', 
            torch.from_numpy(kp).to(torch.float32))
        
        # Projection axis.
        stddev = math.sqrt(1.0 / kp.shape[0])
        self.proj_axes_ = torch.nn.Parameter(
            torch.empty(kp.shape[0], p_num_basis))
        self.proj_axes_.data.uniform_(-stddev, stddev)
        self.proj_biases_ = torch.nn.Parameter(torch.zeros(
            (p_num_basis,), dtype=torch.float32))

        # Multihead attention.
        self.linear_kqv_ = torch.nn.Linear(p_in_features, 3*self.value_size_)
        self.w_out_ = torch.nn.Linear(self.value_size_, p_out_features)

        # PE
        stddev = math.sqrt(1.0 / self.value_size_)
        self.pe_ = torch.nn.Parameter(
            torch.empty(1, p_num_basis, self.value_size_))
        self.pe_.data.uniform_(-stddev, stddev)

    def __compute_convolution__(self,
        p_pc_in,
        p_pc_out,
        p_in_features,
        p_neighborhood):
        """Abstract mehod to implement a convolution.

        Args:
            p_pc_in (Pointcloud): Input point cloud.
            p_pc_out (Pointcloud): Output point cloud.
            p_in_features (tensor nxfi): Input features.
            p_neighborhood (Neighborhood): Input neighborhood.

        Returns:
            tensor mxfo: Output features.
        """

        assert(p_pc_in == p_pc_out)

        # Kernel  point correlations.
        pt_pne = KPPNE.apply(
            p_pc_in.pts_,
            p_pc_out.pts_,
            p_neighborhood.neighbors_,
            self.kernel_pts_,
            self.kp_sigma_,
            self.proj_axes_,
            self.proj_biases_,
            self.norm_neigh_dist_,
            "gauss")

        # Get K Q V
        x = self.linear_kqv_(p_in_features)
        qv = x[:, :self.value_size_*2]
        k = x[:, self.value_size_*2:]

        # Accumulate.
        agg_qv = FeatBasisProj.apply(pt_pne, qv, 
            p_neighborhood.neighbors_, p_neighborhood.start_ids_)  
        agg_v = agg_qv[:, :self.value_size_, :].transpose(1, 2)
        agg_q = agg_qv[:, self.value_size_:, :].transpose(1, 2) + self.pe_

        # Multihead attention.
        head_size = self.value_size_//self.num_heads_
        att_weights = torch.einsum('nkhi,nkhi->nkh',
            agg_q.reshape((x.shape[0], self.num_basis_, self.num_heads_, head_size)),
            k.reshape((x.shape[0], 1, self.num_heads_, head_size)))
        att_weights = torch.nn.functional.softmax(att_weights, 1)
        agg_v = torch.einsum('nkhi,nkhi->nhi',
            agg_v.reshape((x.shape[0], self.num_basis_, self.num_heads_, head_size)), 
            att_weights.reshape((x.shape[0], self.num_basis_, self.num_heads_, 1))).reshape((x.shape[0], self.value_size_))
        conv_results = self.w_out_(agg_v)
        
        return conv_results*self.norm_num_neighs_ 


class MultiHeadAttLayerFactory(IConvLayerFactory):
    """Interface of a layer actory.
    """

    def __init__(self,
                 p_dims,
                 p_num_basis,
                 p_kp_res,
                 p_num_heads):
        """Constructor.

            Args:
                p_dims (int): Number of dimensions.
                p_num_basis (int): Number of basis.
                p_kp_res (string): Kernel points resolution.
                p_num_heads (int): Number of heads.
        """

        # Super class init.
        super(MultiHeadAttLayerFactory, self).__init__(p_dims)

        # Save parameters.
        self.num_basis_ = p_num_basis
        self.kp_res_ = p_kp_res
        self.num_heads_ = p_num_heads

    
    def update_parameters(self, **kwargs):
        """Method to update the parameters of the class.
        """
        if 'num_basis' in kwargs:
            self.num_basis_ = kwargs['num_basis']
        elif 'kp_res' in kwargs:
            self.kp_res_ = kwargs['kp_res']
        elif 'num_heads' in kwargs:
            self.num_heads_ = kwargs['num_heads']


    def __create_conv_layer_imp__(self,
        p_in_features, p_out_features):
        """Abstract method to create a layer.

        Args:
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        Return IConvLayer object.
        """
        return MultiHeadAttLayer(
            self.dims_, p_in_features, p_out_features, 
            self.num_basis_, self.kp_res_, self.num_heads_)