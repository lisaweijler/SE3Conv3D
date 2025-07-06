import numpy as np
import torch

from point_cloud_lib.pc import Pointcloud
from point_cloud_lib.pc import FPSSubSample
from point_cloud_lib.pc import GridSubSample
from point_cloud_lib.pc import KnnNeighborhood
from point_cloud_lib.pc import BQNeighborhood

class PointHierarchy(object):
    """Class to represent a hierarchy of point clouds.
    """

    def __init__(self, 
        p_point_cloud,
        p_num_sub_samples,
        p_subsample_method = "grid_avg",
        **kwargs):
        '''Constructor.
        
        Args:
            p_point_cloud (Pointcloud): Point Cloud.
            p_num_sub_samples (int): Number of sub-sampled point clouds.
            p_subsample_method (string): Sub-sample method.
        '''

        # Initialize variables.
        self.sub_sampled_objs_ = []
        self.pcs_ = [p_point_cloud]
        cur_pc = p_point_cloud

        # Iterate over the sub samples.
        for i in range(p_num_sub_samples):

            # Compute sub-sampled point cloud.
            new_pc, sub_sample_obj = self.__create_sub_sample__(
                cur_pc, p_subsample_method, i, **kwargs)
            self.sub_sampled_objs_.append(sub_sample_obj)
            self.pcs_.append(new_pc)
            cur_pc = new_pc

        # Init neighborhood cache.
        self.neigh_cache_ = {}
            

    def __create_sub_sample__(self, p_point_cloud, p_samp_method, p_id, **kwargs):
        if p_samp_method == "fps":
            samp = FPSSubSample(p_point_cloud, kwargs['fps_ratios'][p_id])
        elif p_samp_method == "grid_avg":
            samp = GridSubSample(p_point_cloud, kwargs['grid_radii'][p_id], False)
        elif p_samp_method == "grid_rnd":
            samp = GridSubSample(p_point_cloud, kwargs['grid_radii'][p_id], True)
        
        new_pts = samp.__subsample_tensor__(p_point_cloud.pts_, "avg")
        new_batch_ids = samp.__subsample_tensor__(p_point_cloud.batch_ids_, "max")
        new_pc = Pointcloud(new_pts, new_batch_ids)
        return new_pc, samp


    def create_neighborhood(self, p_pc_src_id, p_pc_dest_id, p_neigh_method, **kwargs):
        neigh_str = str(p_pc_src_id)+"_"+str(p_pc_dest_id)+"_"+p_neigh_method
        if p_neigh_method == "knn":
            neigh_str = neigh_str + str(kwargs['neihg_k'])
        elif p_neigh_method == "ball_query":
            neigh_str = neigh_str + str(kwargs['bq_radius'])
        if neigh_str in self.neigh_cache_:
            return self.neigh_cache_[neigh_str]
        else:
            if p_neigh_method == "knn":
                neighborhood = KnnNeighborhood(
                    self.pcs_[p_pc_src_id], self.pcs_[p_pc_dest_id], 
                    kwargs['neihg_k'])
            elif p_neigh_method == "ball_query":
                neighborhood = BQNeighborhood(
                    self.pcs_[p_pc_src_id], self.pcs_[p_pc_dest_id], 
                    kwargs['bq_radius'])
                
            self.neigh_cache_[neigh_str] = neighborhood
            return neighborhood
        

    def clear_neigh_cache(self):
        self.neigh_cache_ = {}


    def pool_tensor(self, p_tensor, p_pc_src_id, p_pc_dest_id, p_pool_method):
        assert(p_pc_dest_id-p_pc_src_id == 1)
        return self.sub_sampled_objs_[p_pc_src_id].__subsample_tensor__(p_tensor, p_pool_method)

    
    def upsample_tensor(self, p_tensor, p_pc_src_id, p_pc_dest_id):
        assert(p_pc_src_id-p_pc_dest_id == 1)
        return self.sub_sampled_objs_[p_pc_dest_id].__upsample_tensor__(p_tensor)