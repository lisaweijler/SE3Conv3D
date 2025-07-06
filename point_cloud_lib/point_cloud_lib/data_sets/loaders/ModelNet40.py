import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, Sampler

import point_cloud_lib as pclib

MN40_BASE_AUGMENTATIONS = [
        {
            'name': 'CenterAug',
            'p_apply_extra_tensors': [False]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 0,
            'p_min_angle' : -np.pi/24.0,
            'p_max_angle' : np.pi/24.0,
            'p_apply_extra_tensors': [True]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 2,
            'p_min_angle' : -np.pi/24.0,
            'p_max_angle' : np.pi/24.0,
            'p_apply_extra_tensors': [True]
        },
        {
            'name': 'NoiseAug',
            'p_prob': 1.0,
            'p_stddev' : 0.01,
            'p_apply_extra_tensors': [False]
        },
        {
            'name': 'LinearAug',
            'p_prob': 1.0,
            'p_min_a' : 0.9,
            'p_max_a' : 1.1,
            'p_min_b' : 0.0,
            'p_max_b' : 0.0,
            'p_channel_independent' : True,
            'p_apply_extra_tensors': [False]
        },
        {
            'name': 'MirrorAug',
            'p_prob': 1.0,
            'p_mirror_prob' : 0.5,
            'p_axes' : [True, False, True],
            'p_apply_extra_tensors': [True]
        }
    ]

class ModelNet40_Collate():

    @staticmethod
    def collate(p_batch):
        batch_pts = []
        batch_normals = []
        batch_ids = []
        batch_labels = []
        obj_ids = []
        for cur_iter, cur_batch in enumerate(p_batch):
            batch_pts.append(cur_batch[0])
            batch_normals.append(cur_batch[1])
            batch_ids.append(torch.ones(cur_batch[0].shape[0], dtype=torch.int32)*cur_iter)
            batch_labels.append(cur_batch[2])
            obj_ids.append(cur_batch[3])
            
        batch_pts = torch.cat(batch_pts, 0).to(torch.float32)
        batch_normals = torch.cat(batch_normals, 0).to(torch.float32)
        batch_ids = torch.cat(batch_ids, 0).to(torch.int32)
        batch_labels = torch.from_numpy(np.array(batch_labels)).to(torch.int64)
        
        return batch_pts, batch_normals, batch_ids, batch_labels, obj_ids
    
    
class ModelNet40DS(Dataset):
    """ModelNet40 data set.
    """

    def __init__(self,
                 p_data_folder,
                 p_augmentation_cfg,
                 p_num_pts = 1024,
                 p_split = "train",
                 p_create_tmp_file = True,
                 p_use_coords_as_features = True):
        """Constructor.

            Args:
                p_data_folder (string): Data folder path.
                p_augmentation_cfg (list of dict): List of dictionaries with
                    the different configurations for the data augmentation
                    techniques.
                p_num_pts (int): Number of points.
                p_split (string): Data split used.
                p_create_tmp_file (bool): Boolean that indicates if the object
                    creates a temporal file for future fast loadings.
                p_use_coords_as_features (bool): Boolean that indicates if the 
                    point coordinates are used as features.
        """

        # Super class init.
        super(ModelNet40DS, self).__init__()

        # Save parameters.
        self.path_ = p_data_folder
        self.num_pts_ = p_num_pts
        self.coords_as_features_ = p_use_coords_as_features

        # Configure the data augmentation pipeline.
        if len(p_augmentation_cfg) > 0:
            self.aug_pipeline_ = pclib.augment.AugPipeline()
            self.aug_pipeline_.create_pipeline(p_augmentation_cfg)
        else:
            self.aug_pipeline_ = None

        # Load class list.
        with open(os.path.join(self.path_, 'modelnet40_shape_names.txt'), 'r') as my_file:
            self.class_names_ = [line.rstrip() for line in my_file]

        tmp_file_path = os.path.join(self.path_, "tmp_"+p_split+"_"+str(self.num_pts_)+".h5")
        if os.path.exists(tmp_file_path):
            hf = h5py.File(tmp_file_path, 'r')
            self.pts_ = hf.get('points')[:,:]
            self.normals_ = hf.get('normals')[:,:]
            self.model_class_ = hf.get('model_class')[:]
            hf.close()
        else:
            # Load the file list.
            with open(os.path.join(self.path_, 'modelnet40_'+p_split+".txt"), 'r') as my_file:
                file_list = [line.rstrip() for line in my_file]

            # Load the models.
            self.pts_ = []
            self.normals_ = []
            self.model_class_ = []
            for cur_model in file_list:
                model_class = '_'.join(cur_model.split('_')[:-1])
                model_data = np.loadtxt(
                    os.path.join(
                        os.path.join(self.path_, model_class),
                        cur_model+".txt"),
                    delimiter=',')[:self.num_pts_, :].astype(np.float32)
                self.pts_.append(model_data[:,0:3])
                self.normals_.append(model_data[:,3:])
                self.model_class_.append(self.class_names_.index(model_class))
            
            self.pts_ = np.array(self.pts_).astype(np.float32)
            self.normals_ = np.array(self.normals_).astype(np.float32)
            self.model_class_ = np.array(self.model_class_).astype(np.int32)

            if p_create_tmp_file:
                hf = h5py.File(tmp_file_path, 'w')
                hf.create_dataset('points', data=self.pts_)
                hf.create_dataset('normals', data=self.normals_)
                hf.create_dataset('model_class', data=self.model_class_)
                hf.close()


    def __len__(self):
        """Get the lenght of the data set.

            Returns:
                (int) Number of models.
        """
        return len(self.pts_)
    

    def increase_epoch_counter(self):
        """Method to increase the epoch counter for user-defined augmentations."""
        if not self.aug_pipeline_ is None:
            self.aug_pipeline_.increase_epoch_counter()
        

    def __getitem__(self, idx):
        """Get item in position idx.

            Args:
                idx (int): Index of the element to return.

            Returns:
                (int) Number of models.
        """
        pts = self.pts_[idx]
        normals = self.normals_[idx]
        model_class = self.model_class_[idx]

        pts = torch.from_numpy(pts).to(torch.float32)
        normals = torch.from_numpy(normals).to(torch.float32)

        if self.aug_pipeline_:
            pts, _, normals = self.aug_pipeline_.augment(pts, [normals])
            normals = normals[0]

        if self.coords_as_features_:
            normals = torch.cat((normals, pts), -1)
            
        return pts, normals, model_class, idx
