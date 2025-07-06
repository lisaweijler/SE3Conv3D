import os
import numpy as np
from tqdm import tqdm
import torch
import trimesh
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing_dataset import AMASSPreLoadDataset

from geometry import (
    batch_rodrigues,
    get_body_model,
    rotation_matrix_to_angle_axis,
)

SOURCE_DATA_PATH = (
    "/caa/Homes01/lweijler/phd/point_clouds/published_repos/SE3Conv3D/data"
)
SAVE_DATA_PATH = "/data/lweijler/SE3Conv3D/dfaust/"


def to_categorical(y, num_classes):
    """1-hot encodes a tensor."""
    new_y = torch.eye(num_classes, device=y.device)[y]
    return new_y


def trimesh_sampling(vertices, faces, count, gt_lbs):
    body_mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces)
    _, sample_face_idx = trimesh.sample.sample_surface_even(body_mesh, count)
    if sample_face_idx.shape[0] != count:
        print("add more face idx to match num_point")
        missing_num = count - sample_face_idx.shape[0]
        add_face_idx = np.random.choice(sample_face_idx, missing_num)
        sample_face_idx = np.hstack((sample_face_idx, add_face_idx))
    r = np.random.rand(count, 2)

    A = vertices[:, faces[sample_face_idx, 0], :]
    B = vertices[:, faces[sample_face_idx, 1], :]
    C = vertices[:, faces[sample_face_idx, 2], :]
    P = (
        (1 - np.sqrt(r[:, 0:1])) * A
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    lbs_w = gt_lbs.cpu().numpy()
    A_lbs = lbs_w[faces[sample_face_idx, 0], :]
    B_lbs = lbs_w[faces[sample_face_idx, 1], :]
    C_lbs = lbs_w[faces[sample_face_idx, 2], :]
    P_lbs = (
        (1 - np.sqrt(r[:, 0:1])) * A_lbs
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B_lbs
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C_lbs
    )

    return P, P_lbs


def sample_points(
    vertices, faces, count, gt_lbs, fix_sample=False, sample_type="trimesh"
):
    assert not fix_sample and sample_type == "trimesh"
    return trimesh_sampling(vertices, faces, count, gt_lbs)


def get_pointcloud(vertices, n_points_surface, points_sigma, gt_lbs, body_model_faces):
    points_surface, points_surface_lbs = sample_points(
        vertices=vertices.cpu().numpy(),
        faces=body_model_faces,
        count=n_points_surface,
        gt_lbs=gt_lbs,
    )

    points_surface = torch.from_numpy(points_surface).to(vertices.device)
    points_surface_lbs = torch.from_numpy(points_surface_lbs).to(vertices.device)

    labels = get_joint_label_merged(points_surface_lbs)

    points_surface += points_sigma * torch.randn(points_surface.shape[1], 3).to(
        vertices.device
    )
    points_label = labels[None, :].repeat(vertices.size(0), 1)

    # fig =
    return points_surface.float(), points_label, points_surface_lbs


def get_joint_label_merged(lbs_weights):
    gt_joint = torch.argmax(lbs_weights, dim=1)
    gt_joint1 = torch.where((gt_joint == 22), 20, gt_joint)
    gt_joint2 = torch.where((gt_joint1 == 23), 21, gt_joint1)
    gt_joint2 = torch.where((gt_joint2 == 10), 7, gt_joint2)
    gt_joint2 = torch.where((gt_joint2 == 11), 8, gt_joint2)

    return gt_joint2


def SMPLX_layer(body_model, betas, translation, motion_pose, rep="6d"):
    bz = body_model.batch_size

    if rep == "rotmat":
        motion_pose_aa = rotation_matrix_to_angle_axis(
            motion_pose.reshape(-1, 3, 3)
        ).reshape(bz, -1)
    else:
        motion_pose = motion_pose.squeeze().reshape(bz, -1)
        motion_pose_aa = motion_pose

    zero_center = torch.zeros_like(translation.reshape(-1, 3).cuda("cuda:6"))
    body_param_rec = {}
    body_param_rec["transl"] = zero_center
    body_param_rec["global_orient"] = motion_pose_aa[:, :3].cuda("cuda:6")
    body_param_rec["body_pose"] = torch.cat(
        [motion_pose_aa[:, 3:66].cuda("cuda:6"), torch.zeros(bz, 6).cuda("cuda:6")],
        dim=1,
    )
    body_param_rec["betas"] = betas.reshape(bz, -1)[:, :10].cuda("cuda:6")

    body_mesh = body_model(return_verts=True, **body_param_rec)
    mesh_rec = body_mesh.vertices
    mesh_j_pose = body_mesh.joints
    return mesh_j_pose, mesh_rec


def preload_data(
    save_dpath, body_model, data_loader, device, num_points, save_plot_data=False
):

    for idx, batch_data in tqdm(enumerate(data_loader)):
        motion_pose_aa = batch_data["rotations"].to(device)

        motion_trans = batch_data["translation"].to(device)
        B, _ = motion_trans.size()

        motion_pose_rotmat = batch_rodrigues(motion_pose_aa.reshape(-1, 3)).reshape(
            B, -1, 3, 3
        )

        betas = batch_data["body_shape"][:, None, :].to(device)

        gt_joints_pos, gt_vertices = SMPLX_layer(
            body_model, betas, motion_trans, motion_pose_rotmat, rep="rotmat"
        )

        pcl_data, label_data, pcl_lbs = get_pointcloud(
            gt_vertices,
            num_points,
            points_sigma=0.001,
            gt_lbs=body_model.lbs_weights,
            body_model_faces=body_model.faces.astype(int),
        )

        torch.save(pcl_data.squeeze(0), save_dpath / f"model_{idx}_pc.pt")
        torch.save(label_data.squeeze(0), save_dpath / f"model_{idx}_labels.pt")

        if save_plot_data:
            if not os.path.exists(save_dpath + "model_plots/"):
                os.makedirs(save_dpath + "model_plots/")
            fig = plt.figure(figsize=[10, 5])
            ax = fig.add_subplot(111, projection="3d")
            d = pcl_data.squeeze(0).cpu().numpy()
            l = label_data.squeeze(0).cpu().numpy()

            ax.scatter3D(d[:, 0], d[:, 1], d[:, 2], s=0.1, c=l, cmap="Dark2")
            plt.savefig(save_dpath + f"model_plots/model_{idx}.png")
            plt.close()
            np.savetxt(save_dpath + f"model_plots/model_{idx}.txt", d)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_points = 15000
    body_model = get_body_model(
        model_type="smpl", gender="male", batch_size=1, device=device
    )

    # preload/preprocess DFAUST dataset
    # dataset_train = AMASSPreLoadDataset(src_data_path=SOURCE_DATA_PATH, is_train=True)
    dataset_test = AMASSPreLoadDataset(src_data_path=SOURCE_DATA_PATH, is_train=False)

    # preload/preprocess DFAUST dataloaders
    # train_loader = DataLoader(
    #     dataset_train,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    test_loader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # save path
    # save_dpath_train = Path(SAVE_DATA_PATH + "train/")
    save_dpath_test = Path(SAVE_DATA_PATH + "test/")

    save_dpath_test.mkdir(parents=True, exist_ok=True)
    # save_dpath_train.mkdir(parents=True, exist_ok=True)

    print(f"## Preprocessing DFAUST train dataset...")
    # preload_data(save_dpath_train, body_model, train_loader, device, num_points)
    print(f"... done")
    print(f"## Preprocessing DFAUST test dataset (MPI LIMITS)...")
    preload_data(save_dpath_test, body_model, test_loader, device, num_points)
    print(f"... done")
