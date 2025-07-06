import os
from pathlib import Path
import sys
import time
import argparse
from einops import repeat
from tqdm import tqdm
import yaml
import glob
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
import point_cloud_lib as pclib

current_milli_time = lambda: time.time() * 1000.0

MAX_NUM_THREADS = 4
GPU_ID = 0

torch.set_num_threads(MAX_NUM_THREADS)


############## DATA LOADERS
def create_data_loaders(p_ds_dict, p_batch_size, p_data_folder):

    if not p_ds_dict["test_aug_file"] == "None":
        aug_test = importlib.import_module(p_ds_dict["test_aug_file"])
    else:
        aug_test = None

    testds = pclib.data_sets.loaders.DFaustDS(
        p_data_folder=p_data_folder,
        p_augmentation_cfg=aug_test.DS_AUGMENTS if not aug_test is None else [],
        p_num_pts=p_ds_dict["num_points"],
        p_split="val",
    )

    testdl = DataLoader(
        testds,
        batch_size=p_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=pclib.data_sets.loaders.DFaust_Collate.collate,
        num_workers=0,
    )

    num_classes = 20
    num_in_feats = 1
    mask_cls = []  # [10, 11, 22]

    return testds, testdl, num_classes, num_in_feats, mask_cls


############## MODEL
def create_model(p_model_dict, p_num_classes, p_num_in_feats, p_param_list):
    # Load model class.
    spec = importlib.util.spec_from_file_location(
        "models",
        "/caa/Homes01/lweijler/phd/point_clouds/published_repos/SE3Conv3D/tasks/SemSeg/seg_models.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, p_model_dict["model"])

    # Create models.
    out_models = []
    for cur_param in p_param_list:
        model = model_class(
            p_num_in_feats=p_num_in_feats,
            p_num_out_classes=p_num_classes,
            p_max_path_drop=p_model_dict["max_drop_path"],
        )
        model.cuda(device=GPU_ID)
        model.load_state_dict(cur_param)
        out_models.append(model)

    # Count parameters.
    param_count = 0
    for p in list(out_models[0].parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        param_count += nn

    return out_models, param_count


############## CREATE HIERARCHY
def create_hierarchy(p_model_dict, p_pts, p_batch_ids, p_features, p_test_dict):
    with torch.no_grad():
        pc = pclib.pc.PointcloudRotEquiv(
            p_pts, p_batch_ids, p_test_dict["RefFrames"], standard_knn=True
        )
        # pc = pclib.pc.Pointcloud(p_pts, p_batch_ids)
        samp = pclib.pc.GridSubSample(pc, p_model_dict["init_subsample"])
        new_pts = samp.__subsample_tensor__(pc.pts_, "avg")
        new_batch_ids = samp.__subsample_tensor__(pc.batch_ids_, "max")
        new_features = samp.__subsample_tensor__(p_features, "avg")
        new_pc = pclib.pc.PointcloudRotEquiv(
            new_pts, new_batch_ids, p_test_dict["RefFrames"]
        )

        hierarchy = pclib.pc.PointHierarchyRotEquiv(
            new_pc,
            len(p_model_dict["grid_subsamples"]),
            "grid_avg",
            grid_radii=p_model_dict["grid_subsamples"],
        )

        levels_radii = [p_model_dict["init_subsample"]] + p_model_dict[
            "grid_subsamples"
        ]

        return pc, hierarchy, new_features, levels_radii


############## MASK VALID POINTS
def mask_valid_points(p_labels, p_mask_cls):
    cur_mask = np.ones_like(p_labels).astype(bool)
    for cur_class in p_mask_cls:
        aux_mask = p_labels != cur_class
        cur_mask = np.logical_and(cur_mask, aux_mask)
    return cur_mask


def validation(p_model_dict, p_model, p_data_loader, p_accum_logits, p_test_dict):

    with torch.no_grad():

        p_model.eval()
        for cur_iter, cur_batch in tqdm(enumerate(p_data_loader)):

            start_batch_time = current_milli_time()
            init_pc, hierarchy, features, lev_radii = create_hierarchy(
                p_model_dict,
                cur_batch[0].cuda(device=GPU_ID),
                cur_batch[2].cuda(device=GPU_ID),
                cur_batch[1].cuda(device=GPU_ID),
                p_test_dict,
            )

            features = repeat(
                features,
                "n d -> (n times) d",
                times=p_test_dict["RefFrames"]["n_frames"],
            )
            pred = p_model(hierarchy, features, lev_radii, init_pc)
            p_accum_logits[cur_iter] += pred.cpu().numpy()

            end_batch_time = current_milli_time()

            if cur_iter % 50 == 0:
                print(
                    "{:5d} / {:5d} ({:.1f} ms)".format(
                        cur_iter, len(p_data_loader), end_batch_time - start_batch_time
                    )
                )

    print()


############## SAVE PREDICTIONS
def save_results(save_folder, per_class_acc, per_class_iou, mIoU, mAcc):
    # np.savetxt(str(save_folder / "accum_logits.txt"), accum_logits)

    np.savetxt(str(save_folder / "per_class_acc.txt"), per_class_acc)
    np.savetxt(str(save_folder / "per_class_iou.txt"), per_class_iou)

    with open(str(save_folder / ("final_stats_results.txt")), "w") as text_file:
        text_file.write("mAcc: {:.4f} ".format(mAcc) + "\n")
        text_file.write("mIoU: {:.4f} ".format(mIoU) + "\n")


############## MAIN

############## MAIN
if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser(description="Test Sematic segmentation DFaust")
    parser.add_argument(
        "--conf_file",
        default="/caa/Homes01/lweijler/phd/point_clouds/published_repos/SE3Conv3D/tasks/SemSeg/confs/dfaust/dfaust_test.yaml",
        help="Configuration file (default: confs/dfaust/dfaust_test.yaml)",
    )

    parser.add_argument(
        "--data_folder",
        default="/data/lweijler/SE3Conv3D/dfaust",
        help="Path to preprocessed data folder (default: /data/lweijler/SE3Conv3D/dfaust)",
    )
    parser.add_argument("--gpu", type=int, default=3, help="GPU Id (default: 0)")

    args = parser.parse_args()

    # Set the gpu id.
    GPU_ID = args.gpu

    # Parse config file.
    with open(args.conf_file, "r") as f:
        conf_file = yaml.safe_load(f)
    test_dict = conf_file["Testing"]
    dataset_dict = conf_file["Dataset"]
    print()

    results_folder = Path("results/dfaust/")
    results_folder.mkdir(parents=True, exist_ok=True)
    model_paths = ["path to your model dir/model_epoch_149.pth"]
    n_frames_testing = [2]

    for m_path in model_paths:
        for n_frames in n_frames_testing:
            # adapt config
            test_dict = conf_file["Testing"]
            dataset_dict = conf_file["Dataset"]

            if "lieconv" in m_path:
                test_dict["RefFrames"] = {
                    "pca": False,
                    "fixed_axis": False,
                    "n_frames": n_frames,
                }
            else:
                test_dict["RefFrames"] = {
                    "pca": True,
                    "neigh_method": "knn",
                    "neigh_kwargs": {"neigh_k": 16},
                    "fixed_axis": False,
                    "n_frames": n_frames,
                }

            # Create the data loaders.
            start_data_time = current_milli_time()
            testds, testdl, num_cls, num_in_feats, mask_cls = create_data_loaders(
                p_ds_dict=dataset_dict,
                p_batch_size=test_dict["batch_size"],
                p_data_folder=args.data_folder,
            )
            end_data_time = current_milli_time()
            print(
                "### Data Loaded ({:d} models test) {:.2f} s".format(
                    len(testds), (end_data_time - start_data_time) / 1000.0
                )
            )

            # Load the model parameters.
            param_list = []
            list_files = glob.glob(m_path)
            for cur_file in list_files:
                dictionary = torch.load(cur_file, map_location="cpu")
                param_list.append(dictionary["params_dict"])
                model_dict = dictionary["model_dict"]
            print()

            # Create the model.
            start_model_time = current_milli_time()
            models, param_count = create_model(
                model_dict, num_cls, num_in_feats, param_list
            )
            end_model_time = current_milli_time()
            print(
                "### Model {:d} {:s} Created ({:d} params) {:.2f} ms".format(
                    len(models),
                    model_dict["model"],
                    param_count,
                    end_model_time - start_model_time,
                )
            )

            # Initialize logits.
            accum_logits = []

            point_labels = []
            point_pos = []

            for cur_batch in testdl:

                accum_logits.append(
                    np.zeros((cur_batch[0].shape[0], num_cls), dtype=np.float32)
                )

                point_labels.append(cur_batch[3].cpu().numpy())
                point_pos.append(cur_batch[0].cpu().numpy())

            # Iterate over the epochs.
            for cur_epoch in range(test_dict["num_epochs"]):
                print()
                print(
                    "### EPOCH {:4d} / {:4d}".format(cur_epoch, test_dict["num_epochs"])
                )

                start_epoch_time = current_milli_time()
                validation(
                    model_dict,
                    models[cur_epoch % len(models)],
                    testdl,
                    accum_logits,
                    test_dict,
                )
                end_epoch_time = current_milli_time()

                testds.increase_epoch_counter()

                print(
                    "Time {:.2f} min".format(
                        (end_epoch_time - start_epoch_time) / 60000.0
                    )
                )

            # Validate.
            accum_metric = pclib.metrics.SemSegMetrics(num_cls, mask_cls)

            for cur_iter in range(len(accum_logits)):
                n_samples = int(
                    accum_logits[cur_iter].shape[0] / dataset_dict["num_points"]
                )
                for i in range(n_samples):
                    mask_valid = mask_valid_points(
                        point_labels[cur_iter][
                            i
                            * dataset_dict["num_points"] : (i + 1)
                            * dataset_dict["num_points"]
                        ],
                        mask_cls,
                    )
                    cur_logits = accum_logits[cur_iter][
                        i
                        * dataset_dict["num_points"] : (i + 1)
                        * dataset_dict["num_points"],
                        :,
                    ][mask_valid]
                    cur_labels = point_labels[cur_iter][
                        i
                        * dataset_dict["num_points"] : (i + 1)
                        * dataset_dict["num_points"]
                    ][mask_valid]
                    accum_metric.update_metrics(cur_logits, cur_labels)

            per_class_iou = accum_metric.per_class_iou()
            per_class_acc = accum_metric.per_class_acc()

            print()
            for i in range(per_class_iou.shape[0]):
                if len(mask_cls) > 0:
                    cur_class = i + len(mask_cls)
                else:
                    cur_class = i

                print(
                    "{:15s}:   {:5.2f}     {:5.2f}   ({:9d})".format(
                        testds.class_names_[cur_class],
                        per_class_iou[i],
                        per_class_acc[i],
                        int(accum_metric.accum_gt_[cur_class]),
                    )
                )

            print()
            mIoU = accum_metric.class_mean_iou()
            mAcc = accum_metric.class_mean_acc()

            print("IoU: {:5.2f} | mAcc: {:5.2f}".format(mIoU, mAcc))

            # Save output.

            if dataset_dict["dataset"] == "dfaust":

                print()
                print("### SAVING OUTPUT")

                exp_results_folder = results_folder / (
                    "train-"
                    + Path(m_path).parent.parent.name
                    + "_"
                    + Path(m_path).parent.name
                    + "_test-OOD_"
                    + str(n_frames)
                    + "_frames_1024pts_rebuttal"
                )
                exp_results_folder.mkdir(parents=False, exist_ok=False)
                save_results(
                    exp_results_folder, per_class_acc, per_class_iou, mIoU, mAcc
                )
