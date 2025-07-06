from point_cloud_lib.pc import FPSSubSample
from point_cloud_lib.pc import GridSubSample
from point_cloud_lib.pc import PointcloudRotEquiv
from point_cloud_lib.pc import PointHierarchy


class PointHierarchyRotEquiv(PointHierarchy):
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

        super(PointHierarchyRotEquiv, self).__init__(p_point_cloud,
                        p_num_sub_samples,
                        p_subsample_method = p_subsample_method,
                        **kwargs)
            

    def __create_sub_sample__(self, p_point_cloud, p_samp_method, p_id, **kwargs):
        """Overload from baseclass. 
        """
        if p_samp_method == "fps":
            samp = FPSSubSample(p_point_cloud, kwargs['fps_ratios'][p_id])
        elif p_samp_method == "grid_avg":
            samp = GridSubSample(p_point_cloud, kwargs['grid_radii'][p_id], False)
        elif p_samp_method == "grid_rnd":
            samp = GridSubSample(p_point_cloud, kwargs['grid_radii'][p_id], True)
        
        new_pts = samp.__subsample_tensor__(p_point_cloud.pts_, "avg")
        new_batch_ids = samp.__subsample_tensor__(p_point_cloud.batch_ids_, "max")
        
        new_pc = PointcloudRotEquiv(new_pts, new_batch_ids, p_point_cloud.local_frames_config_)
        return new_pc, samp



        

