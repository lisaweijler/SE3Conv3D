import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, Sampler

import point_cloud_lib as pclib

SCANNET_BASE_AUGMENTATIONS = [
        {
            'name': 'CenterAug',
            'p_apply_extra_tensors': [False, False, False]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 2,
            'p_min_angle' : 0.0,
            'p_max_angle' : 2.0*np.pi,
            'p_apply_extra_tensors': [True, False, False]
        },
        {
            'name': 'CropAug',
            'p_prob' : 1.0,
            'p_min_crop_size' : 3.0,
            'p_max_crop_size' : 5.0, 
            'p_apply_extra_tensors' : [True, True, True]
        },
        {
            'name': 'CenterAug',
            'p_apply_extra_tensors': [False, False, False]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 0,
            'p_min_angle' : -np.pi/24.0,
            'p_max_angle' : np.pi/24.0,
            'p_apply_extra_tensors': [True, False, False]
        },
        {
            'name': 'RotationAug',
            'p_prob': 1.0,
            'p_axis' : 1,
            'p_min_angle' : -np.pi/24.0,
            'p_max_angle' : np.pi/24.0,
            'p_apply_extra_tensors': [True, False, False]
        },
        {
            'name': 'LinearAug',
            'p_prob': 1.0,
            'p_min_a' : 0.9,
            'p_max_a' : 1.1,
            'p_min_b' : 0.0,
            'p_max_b' : 0.0,
            'p_channel_independent' : True,
            'p_apply_extra_tensors': [False, False, False]
        },
        {
            'name': 'MirrorAug',
            'p_prob': 1.0,
            'p_mirror_prob' : 0.5,
            'p_axes' : [True, True, False],
            'p_apply_extra_tensors': [True, False, False]
        }
    ]

SCANNET_BASE_COLOR_AUGMENTATIONS = [
        {
            'name': 'LinearAug',
            'p_prob': 1.0,
            'p_min_a' : 0.8,
            'p_max_a' : 1.2,
            'p_min_b' : -0.2,
            'p_max_b' : 0.2,
            'p_channel_independent' : False,
            'p_apply_extra_tensors': []
        },
        {
            'name': 'LinearAug',
            'p_prob': 1.0,
            'p_min_a' : 1.0,
            'p_max_a' : 1.0,
            'p_min_b' : -0.2,
            'p_max_b' : 0.2,
            'p_channel_independent' : True,
            'p_apply_extra_tensors': []
        },
    ]

class ScanNet_Collate():

    @staticmethod
    def collate(p_batch):
        batch_pts = []
        batch_features = []
        batch_ids = []
        batch_segments = []
        batch_labels = []
        batch_instances = []
        batch_valid_pts_ids = []
        scene_ids = []
        cur_batch_id = 0
        prev_mixed = False
        for _, cur_batch in enumerate(p_batch):

            batch_pts.append(cur_batch[0])
            batch_features.append(cur_batch[1])
            batch_ids.append(torch.ones(cur_batch[0].shape[0], dtype=torch.int32)*cur_batch_id)
            
            if not cur_batch[2] is None:
                batch_segments.append(cur_batch[2])
            
            if not cur_batch[3] is None:
                batch_labels.append(cur_batch[3])
            else:
                batch_labels.append(torch.zeros((cur_batch[0].shape[0],), dtype=torch.int32))

            if not cur_batch[4] is None:
                batch_instances.append(cur_batch[4])
                
            scene_ids.append(cur_batch[5])

            batch_valid_pts_ids.append(cur_batch[6])

            if not cur_batch[7] or prev_mixed:
                cur_batch_id += 1
                prev_mixed = False
            else:
                prev_mixed = True
                
        batch_pts = torch.cat(batch_pts, 0).to(torch.float32)
        batch_features = torch.cat(batch_features, 0).to(torch.float32)
        batch_ids = torch.cat(batch_ids, 0).to(torch.int32)
        batch_labels = torch.cat(batch_labels, 0).to(torch.int64)
        batch_valid_pts_ids = torch.cat(batch_valid_pts_ids, 0).to(torch.int32)

        return_list = [batch_pts, batch_features, batch_ids]
        if len(batch_segments) > 0:
            batch_segments = torch.cat(batch_segments, 0).to(torch.int64)
            return_list.append(batch_segments)
        return_list.append(batch_labels)
        if len(batch_instances) > 0:
            batch_instances = torch.cat(batch_instances, 0).to(torch.int32)
            return_list.append(batch_instances)
        return_list = return_list + [scene_ids, batch_valid_pts_ids]

        return return_list
    
    
class ScanNetDS(Dataset):
    """ScanNet data set.
    """

    def __init__(self,
                 p_data_folder,
                 p_dataset,
                 p_augmentation_cfg,
                 p_augmentation_color_cfg,
                 p_prob_mix3d = 0.8,
                 p_split = "train",
                 p_load_segments = False,
                 p_return_instances = False,
                 p_pt_coords_as_feats = False,
                 p_scale_pt_feats = 1.0/5.0):
        """Constructor.

            Args:
                p_data_folder (string): Data folder path.
                p_dataset (string): Data set.
                p_augmentation_cfg (list of dict): List of dictionaries with
                    the different configurations for the data augmentation
                    techniques.
                p_augmentation_color_cfg (list of dict): List of dictionaries with
                    the different configurations for the data augmentation
                    techniques used for the colors.
                p_prob_mix3d (float): Probability of mixing scenes.
                p_split (string): Data split used.
                p_load_segments (bool): Boolean that indicates if segment ids are loaded.
                p_return_instances (bool): Boolean that indicates if the instaces ids are returned.
                p_scale_pt_feats (float): Scale used for the point features.
        """

        # Super class init.
        super(ScanNetDS, self).__init__()

        # Save parameters.
        self.path_ = p_data_folder
        self.dataset_ = p_dataset
        self.split_ = p_split
        self.prob_mix_3d_ = p_prob_mix3d
        self.load_segments_ = p_load_segments
        self.data_aug_enabled_ = True
        self.return_instances_ = p_return_instances
        self.pt_coords_as_feats_ = p_pt_coords_as_feats
        self.scale_pt_feats_ = p_scale_pt_feats

        # Configure the data augmentation pipeline.
        if len(p_augmentation_cfg) > 0:
            self.aug_pipeline_ = pclib.augment.AugPipeline()
            self.aug_pipeline_.create_pipeline(p_augmentation_cfg)
        else:
            self.aug_pipeline_ = None
        if len(p_augmentation_color_cfg) > 0:
            self.aug_pipeline_color_ = pclib.augment.AugPipeline()
            self.aug_pipeline_color_.create_pipeline(p_augmentation_color_cfg)
        else:
            self.aug_pipeline_color_ = None

        # Load class list.
        if self.dataset_ == "scannet20":
            self.class_names_ = [
                'unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
        elif self.dataset_ == "scannet200":
            self.class_names_ = [
                'unannotated', 'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
                'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
                'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
                'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
                'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
                'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
                'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
                'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
                'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
                'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
                'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']

        # Masked classes.
        self.mask_classes_ = [0]
        if self.dataset_ == "scannet200" and not "train" in p_split:
            only_train = ['bicycle', 'storage container', 'candle', 'guitar case', 'purse', 'alarm clock', 'music stand', 'cd case', 
                        'structure', 'storage organizer', 'luggage']
            for cur_class in only_train:
                aux_idx = self.class_names_.index(cur_class)
                self.mask_classes_.append(aux_idx)

        # Load scenes.
        self.file_list_ = []
        self.model_list_ = []
        if p_split == "train+val":
            self.load_split(p_data_folder, "train")
            self.load_split(p_data_folder, "val")
        else:
            self.load_split(p_data_folder, p_split)

        # Load color stats.
        with open(p_data_folder+"/color_stats.txt", 'r') as my_file:
            lines = my_file.readlines()
            mean_line = lines[0].rstrip().split(',')
            std_line = lines[1].rstrip().split(',')
        self.color_mean_ = np.array([float(mean_line[0]), float(mean_line[1]), float(mean_line[2])])
        self.color_std_ = np.array([float(std_line[0]), float(std_line[1]), float(std_line[2])])

        # Label stats.
        if self.dataset_ == "scannet20":
            label_stats_file = "label_20_stats.txt"
        elif self.dataset_ == "scannet200":
            label_stats_file = "label_200_stats.txt"
        with open(p_data_folder+"/"+label_stats_file, 'r') as my_file:
            lines = my_file.readlines()
            self.label_stats_ = np.array([float(i.rstrip()) for i in lines]).astype(np.float32)


    def load_split(self, p_data_folder, p_split):

        # Load the file list.
        cur_file_list = []
        with open(p_data_folder+"/scannet_"+p_split+".txt", 'r') as my_file:
            for cur_line in my_file:
                cur_file_list.append(cur_line.rstrip())
                self.file_list_.append(cur_line.rstrip())

        # Load the models.
        for cur_model in cur_file_list:
            loaded_model = np.load(p_data_folder+"/"+p_split+"/"+cur_model+".npz")
            cur_pts = loaded_model['points']
            cur_normals = loaded_model['normals']
            cur_rgb = loaded_model['colors']

            if self.load_segments_:
                segments_file = np.load(p_data_folder+"/segments/"+cur_model+"_seg.npz")
                _, cur_seg_ids = np.unique(segments_file['segments'], return_inverse=True)

            if p_split != "test":
                if self.dataset_ == "scannet20":
                    cur_label = loaded_model['labels_20']
                elif self.dataset_ == "scannet200":
                    cur_label = loaded_model['labels_200']

                cur_instances = loaded_model['obj_instance']

                if self.load_segments_:
                    self.model_list_.append((cur_pts, cur_normals, cur_rgb, cur_seg_ids, cur_label, cur_instances))
                else:
                    self.model_list_.append((cur_pts, cur_normals, cur_rgb, cur_label, cur_instances))
            else:
                if self.load_segments_:
                    self.model_list_.append((cur_pts, cur_normals, cur_rgb, cur_seg_ids))
                else:
                    self.model_list_.append((cur_pts, cur_normals, cur_rgb))


    def get_num_pts(self, p_room_idx):
        """Get the numbers of points per room.
            
            Args:
                p_room_idx (int): Room index.

            Returns:
                (int) Number of points.
        """
        assert(p_room_idx < len(self.model_list_))
        return self.model_list_[p_room_idx][0].shape[0]


    def __len__(self):
        """Get the lenght of the data set.

            Returns:
                (int) Number of models.
        """
        return len(self.model_list_)
    

    def increase_epoch_counter(self):
        """Method to increase the epoch counter for user-defined augmentations."""
        if not self.aug_pipeline_ is None:
            self.aug_pipeline_.increase_epoch_counter()
        if not self.aug_pipeline_color_ is None:
            self.aug_pipeline_color_.increase_epoch_counter()

    
    def reset_epoch_counter(self):
        """Method to reset the epoch counter for user-defined augmentations."""
        if not self.aug_pipeline_ is None:
            self.aug_pipeline_.reset_epoch_counter()
        if not self.aug_pipeline_color_ is None:
            self.aug_pipeline_color_.reset_epoch_counter()


    def enale_data_augmentations(self, p_enable):
        """Method to enable disable data augmentations"""
        self.data_aug_enabled_ = p_enable
            

    def __getitem__(self, idx):
        """Get item in position idx.

            Args:
                idx (int): Index of the element to return.

            Returns:
                (int) Number of models.
        """
        cur_model = self.model_list_[idx]

        cur_pts = torch.from_numpy(cur_model[0][:,:3]).to(torch.float32)
        cur_normals = torch.from_numpy(cur_model[1][:,:3]).to(torch.float32)
        cur_rgb = torch.from_numpy((cur_model[2][:,:3] - self.color_mean_.reshape((1,-1)))/\
                self.color_std_.reshape((1,-1))).to(torch.float32)
        valid_id_pts = torch.arange(0, cur_rgb.shape[0])
        
        if self.load_segments_:
            cur_segments = torch.from_numpy(cur_model[3]).to(torch.int32).reshape((-1,))
            next_id = 4
        else:
            cur_segments = None
            next_id = 3

        cur_instances = None
        cur_label = None
        if self.split_ != "test":
            cur_label = torch.from_numpy(cur_model[next_id]).to(torch.int32).reshape((-1,))
            if self.return_instances_:
                cur_instances = torch.from_numpy(cur_model[next_id+1]).to(torch.int32).reshape((-1,))

        # Data augmentation.
        if self.data_aug_enabled_:
            if self.aug_pipeline_:
                add_tensors = [cur_normals, cur_rgb]
                if self.load_segments_: 
                    add_tensors.append(cur_segments)
                if self.split_ != "test":
                    add_tensors.append(cur_label)
                    if self.return_instances_:
                        add_tensors.append(cur_instances)
                cur_pts, params_aug, tensors = self.aug_pipeline_.augment(
                    cur_pts, add_tensors)
                cur_normals = tensors[0]
                cur_rgb = tensors[1]
                if self.load_segments_: 
                    cur_segments = tensors[2]
                    offset_id = 1
                else:
                    offset_id = 0
                if self.split_ != "test":
                    cur_label = tensors[2+offset_id]
                    if self.return_instances_:
                        cur_instances = tensors[3+offset_id]

                for cur_aug_type, cur_aug_params in params_aug:
                    if cur_aug_type == "CropPtsAug":
                        valid_id_pts = valid_id_pts[cur_aug_params]
                    elif cur_aug_type == "CropBoxAug":
                        valid_id_pts = valid_id_pts[cur_aug_params[0]]

            if self.aug_pipeline_color_:
                cur_rgb, _, _ = self.aug_pipeline_color_.augment(cur_rgb, [])

        cur_point_features = torch.cat((cur_normals, cur_rgb), -1)
        if self.pt_coords_as_feats_:
            cur_point_features = torch.cat((cur_point_features, cur_pts*self.scale_pt_feats_), -1)


        return cur_pts.contiguous().to(torch.float32), \
            cur_point_features.contiguous().to(torch.float32), cur_segments, \
            cur_label, cur_instances, idx, valid_id_pts, \
            torch.rand(1).item() < self.prob_mix_3d_


class ScanNetMaxPtsSampler(Sampler):

    def __init__(self, 
            p_num_batches,
            p_max_points_x_batch, 
            p_data_set,
            p_max_scene_pts = 0,
            p_pts_crop_ratio = 1.0):

        self.num_batches_ = p_num_batches
        self.max_points_x_batch_ = p_max_points_x_batch
        self.data_set_ = p_data_set

        self.room_pts_ = []
        for cur_room_idx in np.arange(len(p_data_set)):
            cur_num_pts = self.data_set_.get_num_pts(cur_room_idx)
            max_num_pts = p_max_scene_pts if p_max_scene_pts > 0 else cur_num_pts
            max_num_pts = min(max_num_pts, int(cur_num_pts * p_pts_crop_ratio))
            self.room_pts_.append(max_num_pts)

        self.list_room_ids_1_ = [i for i in range(len(p_data_set))]
        self.list_room_ids_2_ = [i for i in range(len(p_data_set))]


    def __iter__(self):
        batch_list = []
        
        for cur_batch in range(self.num_batches_):

            sel_room = np.random.randint(len(self.list_room_ids_1_))
            cur_room_idx = self.list_room_ids_1_[sel_room]
            self.list_room_ids_1_.remove(cur_room_idx)
            accum_num_pts = self.room_pts_[cur_room_idx]
            accum_batch_list = [cur_room_idx]

            if len(self.list_room_ids_1_) == 0:
                self.list_room_ids_1_ = self.list_room_ids_2_
                self.list_room_ids_2_ = [i for i in range(len(self.data_set_))]

            enough_pcs = False
            while not enough_pcs:
                left_num_pts = self.max_points_x_batch_ - accum_num_pts
                
                mask_valid_rooms = np.array([False for i in range(len(self.data_set_))])
                mask_valid_rooms[np.array(self.list_room_ids_1_)] = True
                mask_valid_rooms[np.array(self.room_pts_) >= left_num_pts] = False
                list_1 = True
                
                if np.sum(mask_valid_rooms) == 0:

                    mask_valid_rooms[np.array(self.list_room_ids_2_)] = True
                    mask_valid_rooms[np.array(self.room_pts_) >= left_num_pts] = False
                    list_1 = False

                if np.sum(mask_valid_rooms) > 0:
                    valid_room_idxs = np.arange(len(self.data_set_))[mask_valid_rooms]
                    sel_room = np.random.randint(len(valid_room_idxs))
                    cur_room_idx = valid_room_idxs[sel_room]

                    accum_batch_list.append(cur_room_idx)
                    accum_num_pts += self.room_pts_[cur_room_idx]

                    if list_1:
                        self.list_room_ids_1_.remove(cur_room_idx)
                        if len(self.list_room_ids_1_) == 0:
                            self.list_room_ids_1_ = self.list_room_ids_2_
                            self.list_room_ids_2_ = [i for i in range(len(self.data_set_))]
                    else:
                        self.list_room_ids_2_.remove(cur_room_idx)
                        if len(self.list_room_ids_2_) == 0:
                            [i for i in range(len(self.data_set_))]

                else:
                    enough_pcs = True

                if abs(self.max_points_x_batch_ - accum_num_pts) < 50000:
                    enough_pcs = True

            batch_list.append(accum_batch_list)

        return iter(batch_list)


    def __len__(self):
        return self.num_batches_