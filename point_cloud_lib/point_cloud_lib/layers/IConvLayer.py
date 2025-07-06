from abc import ABC, abstractmethod
import math
import torch

from point_cloud_lib.layers import PreProcessModule
from point_cloud_lib.pc import KnnNeighborhood, BQNeighborhood

class IConvLayer(PreProcessModule, ABC):
    """Interface of layer for a point convolution.
    """

    def __init__(self,
        p_dims,
        p_in_features, 
        p_out_features):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        """

        # Super class init.
        super(IConvLayer, self).__init__()

        # Save params.
        self.dims_ = p_dims
        self.feat_input_size_ = p_in_features
        self.feat_output_size_ = p_out_features

        # Create the normalization parameters.
        self.register_buffer('norm_neigh_dist_', 
                             torch.tensor(0, dtype=torch.float32))
        self.register_buffer('norm_num_neighs_', 
                             torch.tensor(0, dtype=torch.float32))


    @abstractmethod
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
        pass


    def forward(self, 
        p_pc_in,
        p_pc_out,
        p_in_features,
        p_neighborhood):
        """Forward pass.

        Args:
            p_pc_in (Pointcloud): Input point cloud.
            p_pc_out (Pointcloud): Output point cloud.
            p_in_features (tensor nxfi): Input features.
            p_neighborhood (Neighborhood): Input neighborhood.

        Returns:
            tensor mxfo: Output features.
        """ 
        # Save parameters.
        if self.pre_process_:
            
            with torch.no_grad():

                # Update the normalization neighborhood distance.
                if isinstance(p_neighborhood, BQNeighborhood):
                    new_norm_neigh_dist = torch.tensor(
                        1.0/p_neighborhood.radius_, dtype=torch.float32)
                elif isinstance(p_neighborhood, KnnNeighborhood):
                    diff_xyz = p_pc_in.pts_[p_neighborhood.neighbors_[:,1],:] \
                        - p_pc_out.pts_[p_neighborhood.neighbors_[:,0],:]
                    dist_xyz = torch.mean(torch.sqrt(torch.sum(diff_xyz**2, -1))).item()
                    new_norm_neigh_dist = torch.tensor(
                        1.0/(2.*dist_xyz), dtype=torch.float32)
                    
                self.norm_neigh_dist_ = 0.9*self.norm_neigh_dist_ + 0.1*new_norm_neigh_dist

                # Update the normalization of number of neighbors.
                new_norm_num_neighs = torch.tensor(
                    p_neighborhood.start_ids_.shape[0]/\
                    p_neighborhood.neighbors_.shape[0], dtype=torch.float32)
                self.norm_num_neighs_ = 0.9*self.norm_num_neighs_ + 0.1*new_norm_num_neighs
                    
        # Compute the convolution.
        return self.__compute_convolution__(
            p_pc_in,
            p_pc_out,
            p_in_features,
            p_neighborhood)


class IConvLayerFactory(ABC):
    """Interface of a layer actory.
    """

    def __init__(self,
                 p_dims):
        """Constructor.

            Args:
                p_dims (int): Number of dimensions.
        """

        # Super class init.
        super(IConvLayerFactory, self).__init__()

        # Save parameters.
        self.dims_ = p_dims

        # Initialize the convolution list.
        self.conv_list_ = []

    
    def update_parameters(self, **kwargs):
        """Method to update the parameters of the class.
        """
        pass


    @abstractmethod
    def __create_conv_layer_imp__(self,
        p_in_features, p_out_features):
        """Abstract method to create a layer.

        Args:
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        Return IConvLayer object.
        """
        pass


    def create_conv_layer(self,
        p_in_features, p_out_features):
        """Method to create a layer.

        Args:
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        Return IConvLayer object.
        """
        conv = self.__create_conv_layer_imp__(
            p_in_features, p_out_features)
        self.conv_list_.append(conv)
        return conv