import torch
import numpy as np
from einops import repeat, rearrange
import hashlib

from torch_scatter import scatter_add


from point_cloud_lib.layers import IConvLayerFactory, create_pts_icosphere, PNEConvLayer
from point_cloud_lib.custom_ops import FeatBasisProj, LinearPNE, KPPNE
from point_cloud_lib.pc import change_direction_to_local_frame, all_index_combinations, get_relative_rot

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


class PNEConvLayerRotEquiv(PNEConvLayer):
    """Point convolution with point neighborhood embeddings.
    """

    rot_tensor_cache = {}
    rel_rot_type = '6D'

    @staticmethod
    def empty_rot_tenors_cache():
        PNEConvLayerRotEquiv.rot_tensor_cache = {}
        

    @staticmethod
    def get_rot_tenors(p_pc_in,
        p_pc_out,
        p_neighborhood,
        radius):

        with torch.no_grad():
            rel_pt = (p_pc_in.pts_[p_neighborhood.neighbors_[:,1],:] - \
                    p_pc_out.pts_[p_neighborhood.neighbors_[:,0],:])*radius
            # hash the tensors
            key = hashlib.sha256(rel_pt.cpu().numpy().tobytes()).hexdigest()

            if key not in PNEConvLayerRotEquiv.rot_tensor_cache:

                # get represneation of rel_pt in pc_out reference frames       
                rel_pt_local_frame_pc_out = change_direction_to_local_frame(direction_vector=rel_pt,
                                                                            ref_frames=p_pc_out.local_frames_[p_neighborhood.neighbors_[:,0],:,:])
                rel_pt_local_frame_pc_out = repeat(rel_pt_local_frame_pc_out, 'n k d -> n (k times) d', times=p_pc_in.n_frames_)

                    
                # get rel orientation beteen p_pc_in frames and p_pc_out frames
                rel_orientation = get_relative_rot(frames_A=p_pc_out.local_frames_[p_neighborhood.neighbors_[:,0],:,:],
                                                frames_B=p_pc_in.local_frames_[p_neighborhood.neighbors_[:,1],:,:],
                                                return_representation=PNEConvLayerRotEquiv.rel_rot_type)                                                                     
                # concate features
                rel_pts_rel_orient = torch.cat((rel_pt_local_frame_pc_out, rel_orientation), dim=-1) # n x (n_framesA*n_frames_B) x 9
                #rel_pts_rel_orient = rel_pt_local_frame_pc_out
                n_frames_combinations = rel_pts_rel_orient.shape[1]

                rel_pts_rel_orient = rearrange(rel_pts_rel_orient,  'n k d -> (n k) d') 
                
                ####### update neighbor indices & features
                    
                neighbs = repeat(p_neighborhood.neighbors_, 'n d -> (n times) d', times=n_frames_combinations)
                # multiply with n_frames_in on right[:,1] on n_frames_out on left [:,0]
                neighbs[:,0] = neighbs[:,0]*p_pc_out.n_frames_
                neighbs[:,1] = neighbs[:,1]*p_pc_in.n_frames_
                # create, repeat and add combination matrix between frames
                frame_combination_idx = all_index_combinations(p_pc_out.n_frames_, 
                                                            p_pc_in.n_frames_, 
                                                            p_pc_in.pts_.device)
                # repeat original number of points times of input point cloud
                frame_combination_idx = repeat(frame_combination_idx, 'n m -> (times n) m', times = rel_pt.shape[0])
                neighbs += frame_combination_idx
                # sort neighbors
                sort_idx = neighbs[:,0].sort().indices
                neighbs = neighbs[sort_idx] # so output frames index is now sorted starting by 0 in the first rows
                # sort relative information to match neighbors matrix
                rel_pts_rel_orient = rel_pts_rel_orient[sort_idx]
                # get start ids
                neighbs_start_ids = scatter_add(
                    torch.ones_like(neighbs[:,1], dtype=torch.int32), 
                    neighbs[:,0], dim=0)#, dim_size=p_neighborhood.start_ids_.shape)
                neighbs_start_ids = torch.cumsum(neighbs_start_ids, 0)


                PNEConvLayerRotEquiv.rot_tensor_cache[key] = {"tensor": rel_pt,
                    "rel_pts_rel_orient": rel_pts_rel_orient,
                    "neighbs": neighbs,
                    "neighbs_start_ids": neighbs_start_ids
                    }
                
            else:
                # check if hash collision occured
                if not torch.all(PNEConvLayerRotEquiv.rot_tensor_cache[key]["tensor"] == rel_pt):
                    raise AssertionError("HASH COLLISON!!")
            
            return PNEConvLayerRotEquiv.rot_tensor_cache[key]
            


        

    def __init__(self,
        p_dims,
        p_in_features, 
        p_out_features,
        p_num_basis,
        p_pne_type):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_num_basis (int): Number of basis.
            p_pne_type (string): Point neighborhood embedding type.
        """

        # Super class init.
        super(PNEConvLayerRotEquiv, self).__init__(
            p_dims,
            p_in_features, 
            p_out_features,
            p_num_basis,
            p_pne_type)
        


    def __compute_convolution__(self,
        p_pc_in,
        p_pc_out,
        p_in_features,
        p_neighborhood):
        """Overload from baseclass
        Abstract mehod to implement a convolution.

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
            ## normally an optimized function:
            # pt_pne = LinearPNE.apply(
            #         p_pc_in.pts_,
            #         p_pc_out.pts_,
            #         p_neighborhood.neighbors_,
            #         self.proj_axes_,
            #         self.proj_biases_,
            #         self.norm_neigh_dist_)
            
            ## here implementation for Rot equiv

        

            rot_tensors_dict = PNEConvLayerRotEquiv.get_rot_tenors(p_pc_in, 
                                                                   p_pc_out, 
                                                                   p_neighborhood,
                                                                   self.norm_neigh_dist_)
  
            # Compute the linear projection.
            pt_pne = torch.matmul(rot_tensors_dict["rel_pts_rel_orient"], self.proj_axes_) + self.proj_biases_

            # apply activation function if specified
            if not self.act_func_ is None:
                pt_pne = self.act_func_(pt_pne)  
        
            # Accumulate.
            result_tensor = FeatBasisProj.apply(pt_pne, p_in_features, 
                rot_tensors_dict["neighbs"], rot_tensors_dict["neighbs_start_ids"])  
        
            # Actual convolution.
            conv_results = torch.einsum('nik,iko->no', result_tensor, self.conv_weights_)

            # divide conv_results by number of in_frames
            conv_results /= p_pc_in.n_frames_
           
            # resulting features have the order of idx 0 - point0 frame 0, indx 1 point0 frame 1 etc..
            return conv_results*self.norm_num_neighs_ #avg num neighbors

              

        # Kernel points PNE
        elif "kp" in self.pne_type_:
            raise Exception("KPNE convolution not implemeted yet for Rot Equiv.")



        # Accumulate.
        result_tensor = FeatBasisProj.apply(pt_pne, p_in_features, 
            p_neighborhood.neighbors_, p_neighborhood.start_ids_)       
        
        # Actual convolution.
        conv_results = torch.einsum('nik,iko->no', result_tensor, self.conv_weights_)
        
        return conv_results*self.norm_num_neighs_ 


class PNEConvLayerRotEquivFactory(IConvLayerFactory):
    """Interface of a layer actory.
    """

    def __init__(self,
                 p_dims,
                 p_num_basis,
                 p_pne_type, 
                 p_rel_rot ='6D'):
        """Constructor.

            Args:
                p_dims (int): Number of dimensions.
                p_num_basis (int): Number of basis.
                p_pne_type (string): Point neighborhood embedding type.
        """

        # Super class init.
        super(PNEConvLayerRotEquivFactory, self).__init__(p_dims)

        # Save parameters.
        self.num_basis_ = p_num_basis
        self.pne_type_ = p_pne_type
        self.rel_rot_ = p_rel_rot

    
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
        PNEConvLayerRotEquiv.rel_rot_type = self.rel_rot_
        return PNEConvLayerRotEquiv(
            self.dims_, p_in_features, p_out_features, 
            self.num_basis_, self.pne_type_)