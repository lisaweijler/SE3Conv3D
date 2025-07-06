from abc import ABC, abstractmethod
import math
import torch
import numpy as np

from torch_scatter import scatter_add, scatter_max

from point_cloud_lib.pc import KnnNeighborhood, BQNeighborhood
from point_cloud_lib.layers import IConvLayer, IConvLayerFactory, create_pts_icosphere
from point_cloud_lib.custom_ops import FeatBasisProj, LinearPNE, KPPNE, TransformNeighConv

def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.
    Arguments:
        x: vector or set of vectors with dimension (n, 3), where n is the
            number of vectors
    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x.
    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """
    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R
    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()
    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    return x @ M


class PNEConvLayer(IConvLayer):
    """Point convolution with point neighborhood embeddings.
    """

    def __init__(self,
        p_dims,
        p_in_features, 
        p_out_features,
        p_num_basis,
        p_pne_type,
        p_aggregation = "add"):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_num_basis (int): Number of basis.
            p_pne_type (string): Point neighborhood embedding type.
            p_aggregation (string): Aggregation method.
        """

        # Super class init.
        super(PNEConvLayer, self).__init__(
            p_dims, p_in_features, p_out_features)
        
        # Save parameters.
        self.num_basis_ = p_num_basis
        self.pne_type_ = p_pne_type
        self.aggregation_ = p_aggregation

        # MLP PNE.
        if "mlp" in self.pne_type_:

            # Create the linear projection axes.
            stddev = math.sqrt(1.0 / p_dims)
            self.proj_axes_ = torch.nn.Parameter(
                torch.empty(p_dims, p_num_basis))
            self.proj_axes_.data.uniform_(-stddev, stddev)
            self.proj_biases_ = torch.nn.Parameter(torch.zeros(
                (p_num_basis,), dtype=torch.float32))
            
            # Activation function.
            if self.pne_type_ == "mlp_relu":
                self.act_func_ = torch.nn.ReLU(inplace=True)
            elif self.pne_type_ == "mlp_gelu":
                self.act_func_ = torch.nn.GELU()
            elif self.pne_type_ == "mlp_sin":
                self.act_func_ = torch.sin
            elif self.pne_type_ == "mlp_softmax":
                self.act_func_ = torch.nn.Softmax(dim=-1)
            elif self.pne_type_ == "mlp_linear":
                self.act_func_ = None

        elif "kp" in self.pne_type_:

            if "double" in self.pne_type_:
                # Create the kernel points.
                kp_scale = 0.35
                kp = create_pts_icosphere(0)*kp_scale
                kp_2 = create_pts_icosphere(1)*kp_scale*2
                kp = np.concatenate((kp, kp_2))
                kp = np.concatenate((kp, np.array([[0.0,0.0,0.0]])))
                kp = kp.astype(np.float32)

                # Define the sigman value.
                if self.pne_type_ == "kp_linear_double":
                    self.kp_sigma_ = 0.2
                elif self.pne_type_ == "kp_gauss_double":
                    self.kp_sigma_ = 0.16
                elif self.pne_type_ == "kp_box_double":
                    self.kp_sigma_ = 1.0
                    
            else:
                # Create the kernel points.
                kp_scale = 0.6
                kp = create_pts_icosphere(0)
                kp = np.concatenate((kp, np.array([[0.0,0.0,0.0]])))
                kp = kp.astype(np.float32)*kp_scale

                # Define the sigman value.
                if self.pne_type_ == "kp_linear":
                    self.kp_sigma_ = 0.4
                elif self.pne_type_ == "kp_gauss":
                    self.kp_sigma_ = 0.3
                elif self.pne_type_ == "kp_box":
                    self.kp_sigma_ = 1.0

            # Random rotate kernel points.
            kp = uniform_random_rotation(kp)            

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

        # Create the convolution weights.
        self.conv_weights_ = torch.nn.Parameter(
            torch.empty(
                p_in_features,
                p_num_basis,
                p_out_features))
        stdv = math.sqrt(1.0 / (p_in_features*p_num_basis))
        self.conv_weights_.data.uniform_(-stdv, stdv)


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

        # Linear PNE
        if "mlp" in self.pne_type_:

            pt_pne = LinearPNE.apply(
                    p_pc_in.pts_,
                    p_pc_out.pts_,
                    p_neighborhood.neighbors_,
                    self.proj_axes_,
                    self.proj_biases_,
                    self.norm_neigh_dist_)
            
            if not self.act_func_ is None:
                pt_pne = self.act_func_(pt_pne) 

        # Kernel points PNE
        elif "kp" in self.pne_type_:

            if self.pne_type_ == "kp_linear":
                corr_funct = "linear"
            elif self.pne_type_ == "kp_gauss":
                corr_funct = "gauss"
            elif self.pne_type_ == "kp_box":
                corr_funct = "box"

            cur_sigma = self.kp_sigma_
            
            pt_pne = KPPNE.apply(
                    p_pc_in.pts_,
                    p_pc_out.pts_,
                    p_neighborhood.neighbors_,
                    self.kernel_pts_,
                    cur_sigma,
                    self.proj_axes_,
                    self.proj_biases_,
                    self.norm_neigh_dist_,
                    corr_funct)

        if self.aggregation_ == "add":

            # Accumulate.
            result_tensor = FeatBasisProj.apply(pt_pne, p_in_features, 
                p_neighborhood.neighbors_, p_neighborhood.start_ids_) 

            # Actual convolution.
            conv_results = torch.einsum('nik,iko->no', result_tensor, self.conv_weights_)

        elif self.aggregation_ == "max":
            
            conv_results = TransformNeighConv.apply(pt_pne, self.conv_weights_, p_in_features, p_neighborhood.neighbors_)
            conv_results = scatter_max(conv_results, p_neighborhood.neighbors_[:,0], dim=0)[0]
        
        return conv_results*self.norm_num_neighs_ 


class PNEConvLayerFactory(IConvLayerFactory):
    """Interface of a layer actory.
    """

    def __init__(self,
                 p_dims,
                 p_num_basis,
                 p_pne_type,
                 p_aggregation = "add"):
        """Constructor.

            Args:
                p_dims (int): Number of dimensions.
                p_num_basis (int): Number of basis.
                p_pne_type (string): Point neighborhood embedding type.
        """

        # Super class init.
        super(PNEConvLayerFactory, self).__init__(p_dims)

        # Save parameters.
        self.num_basis_ = p_num_basis
        self.pne_type_ = p_pne_type
        self.aggregation_ = p_aggregation

    
    def update_parameters(self, **kwargs):
        """Method to update the parameters of the class.
        """
        if 'num_basis' in kwargs:
            self.num_basis_ = kwargs['num_basis']


    def __create_conv_layer_imp__(self,
        p_in_features, p_out_features):
        """Abstract method to create a layer.

        Args:
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        Return IConvLayer object.
        """
        return PNEConvLayer(
            self.dims_, p_in_features, p_out_features, 
            self.num_basis_, self.pne_type_, self.aggregation_)