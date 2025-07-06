import os
import sys
import time
import argparse
from einops import repeat
import yaml
import glob
import importlib
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
import point_cloud_lib as pclib

current_milli_time = lambda: time.time() * 1000.0

MAX_NUM_THREADS = 4
GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

torch.set_num_threads(MAX_NUM_THREADS)


############## DATA LOADERS
def create_data_loaders(p_ds_dict):

    if "scannet" in p_ds_dict["dataset"]:

        if not p_ds_dict["test_aug_file"] == "None":
            aug_test = importlib.import_module(p_ds_dict["test_aug_file"])
        else:
            aug_test = None
        if not p_ds_dict["test_aug_color_file"] == "None":
            aug_color_test = importlib.import_module(p_ds_dict["test_aug_color_file"])
        else:
            aug_color_test = None

        testds = pclib.data_sets.loaders.ScanNetDS(
            p_data_folder="/data/databases/scannet_p",
            p_dataset=p_ds_dict["dataset"],
            p_augmentation_cfg=aug_test.DS_AUGMENTS if not aug_test is None else [],
            p_augmentation_color_cfg=(
                aug_color_test.DS_AUGMENTS if not aug_color_test is None else []
            ),
            p_prob_mix3d=0.0,
            p_split=p_ds_dict["split"],
            p_load_segments=True,
        )
        testdl = DataLoader(
            testds,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=pclib.data_sets.loaders.ScanNet_Collate.collate,
            num_workers=0,
        )

        num_classes = 21
        num_in_feats = 3
        mask_classes = [0]
        use_segments = True

    return testds, testdl, num_classes, num_in_feats, mask_classes, use_segments


############## MODEL
def create_model(p_model_dict, p_num_classes, p_num_in_feats, p_param_list):
    # Load model class.
    spec = importlib.util.spec_from_file_location(
        "models",
        "/caa/Homes01/lweijler/phd/point_clouds/point_clouds_nn/Tasks_Rot_Equiv/SemSeg/seg_models.py",
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

        samp = pclib.pc.GridSubSample(pc, p_model_dict["init_subsample"])
        new_pts = samp.__subsample_tensor__(pc.pts_, "avg")
        new_batch_ids = samp.__subsample_tensor__(pc.batch_ids_, "max")
        new_features = samp.__subsample_tensor__(p_features, "avg")
        new_pc = pclib.pc.PointcloudRotEquiv(
            new_pts, new_batch_ids, p_test_dict["RefFrames"]
        )

        new_features = new_features[:, 3:]

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


############## VAL
def validation(
    p_model_dict, p_model, p_data_loader, p_accum_logits, p_updated_pts, p_test_dict
):

    with torch.no_grad():

        p_model.eval()
        for cur_iter, cur_batch in enumerate(p_data_loader):

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
            p_accum_logits[cur_iter][cur_batch[-1].numpy()] += pred.cpu().numpy()

            p_updated_pts[cur_iter][cur_batch[-1].numpy()] = 0

            end_batch_time = current_milli_time()

            if cur_iter % 50 == 0:
                print(
                    "{:5d} / {:5d} ({:.1f} ms)".format(
                        cur_iter, len(p_data_loader), end_batch_time - start_batch_time
                    )
                )

    print()


############## SAVE PREDICTIONS
from scannet_io import (
    save_scannet20_scene_colors,
    save_scannet20_scene_rnd_colors,
    save_scannet20_scene_labels,
)


def save_predictions_scannet20(p_folders, p_scene_name, p_pts, p_labels):
    save_scannet20_scene_colors(
        os.path.join(p_folders[0], p_scene_name + ".txt"), p_pts, p_labels
    )
    save_scannet20_scene_labels(
        os.path.join(p_folders[1], p_scene_name + ".txt"), p_labels
    )


############## MAIN
if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser(description="Test Sematic segmentation")
    parser.add_argument(
        "--conf_file",
        default="/caa/Homes01/lweijler/phd/point_clouds/point_clouds_nn/Tasks_Rot_Equiv/SemSeg/confs/cvpr24/eccv_resubmission/scannet20_test_rot_I_SO2.yaml",
        help="Configuration file (default: confs/modelnet40.yaml)",
    )
    parser.add_argument(
        "--saved_model",
        default="/caa/Homes01/lweijler/phd/point_clouds/point_clouds_nn/Tasks_Rot_Equiv/SemSeg/logs/scannet20_RotEq_pca_I_005_newsetup/20240305-150158/best.pth",
        help="Saved model (default: model.pth)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU Id (default: 0)")
    parser.add_argument("--no_metrics", dest="metrics", action="store_false")
    parser.add_argument("--save_output", dest="save_output", action="store_true")
    args = parser.parse_args()

    # Set the gpu id.
    GPU_ID = args.gpu

    # Parse config file.
    with open(args.conf_file, "r") as f:
        conf_file = yaml.safe_load(f)
    test_dict = conf_file["Testing"]
    dataset_dict = conf_file["Dataset"]
    print()

    # Create the data loaders.
    start_data_time = current_milli_time()
    testds, testdl, num_cls, num_in_feats, mask_cls, use_segments = create_data_loaders(
        p_ds_dict=dataset_dict
    )
    end_data_time = current_milli_time()
    print(
        "### Data Loaded ({:d} models test) {:.2f} s".format(
            len(testds), (end_data_time - start_data_time) / 1000.0
        )
    )

    # Load the model parameters.
    param_list = []
    list_files = glob.glob(args.saved_model)
    for cur_file in list_files:
        dictionary = torch.load(cur_file, map_location="cpu")
        param_list.append(dictionary["params_dict"])
        model_dict = dictionary["model_dict"]
    print()

    # Create the model.
    start_model_time = current_milli_time()
    models, param_count = create_model(model_dict, num_cls, num_in_feats, param_list)
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
    updated_pts = []
    accum_logits = []
    if use_segments:
        segment_ids = []
    else:
        segment_ids = None
    point_labels = []
    point_pos = []
    scene_names = []
    testds.enale_data_augmentations(False)
    for cur_batch in testdl:
        updated_pts.append(np.ones((cur_batch[4].shape[0],), dtype=np.float32))
        accum_logits.append(
            np.zeros((cur_batch[4].shape[0], num_cls), dtype=np.float32)
        )
        if use_segments:
            segment_ids.append(cur_batch[3].cpu().numpy())
            offset_iter = 1
        point_labels.append(cur_batch[offset_iter + 3].cpu().numpy())
        point_pos.append(cur_batch[0].cpu().numpy())
        scene_names.append(testds.file_list_[cur_batch[offset_iter + 4][0]])
    testds.enale_data_augmentations(True)

    # Iterate over the epochs.
    for cur_epoch in range(test_dict["num_epochs"]):
        print()
        print("### EPOCH {:4d} / {:4d}".format(cur_epoch, test_dict["num_epochs"]))

        start_epoch_time = current_milli_time()
        validation(
            model_dict,
            models[cur_epoch % len(models)],
            testdl,
            accum_logits,
            updated_pts,
            test_dict,
        )
        end_epoch_time = current_milli_time()

        testds.increase_epoch_counter()

        print("Time {:.2f} min".format((end_epoch_time - start_epoch_time) / 60000.0))

    # Compute segment logits if needed.
    if use_segments:
        accum_logits_seg = []
        for cur_iter in range(len(accum_logits)):
            cur_logits_torch = torch.from_numpy(accum_logits[cur_iter]).cuda(
                device=GPU_ID
            )
            cur_segments_torch = (
                torch.from_numpy(segment_ids[cur_iter])
                .cuda(device=GPU_ID)
                .to(torch.int64)
            )
            cur_logits_torch = scatter_mean(cur_logits_torch, cur_segments_torch, dim=0)
            cur_logits_torch = cur_logits_torch[cur_segments_torch]
            cur_logits = cur_logits_torch.cpu().numpy()
            accum_logits_seg.append(cur_logits)

    # Validate.
    if args.metrics:

        non_updated_pts = 0
        accum_metric = pclib.metrics.SemSegMetrics(num_cls, mask_cls)
        if use_segments:
            accum_metric_segments = pclib.metrics.SemSegMetrics(num_cls, mask_cls)
        for cur_iter in range(len(accum_logits)):
            non_updated_pts += np.sum(updated_pts[cur_iter])

            mask_valid = mask_valid_points(point_labels[cur_iter], mask_cls)
            cur_logits = accum_logits[cur_iter][mask_valid]
            cur_labels = point_labels[cur_iter][mask_valid]
            accum_metric.update_metrics(cur_logits, cur_labels)

            if use_segments:
                cur_logits = accum_logits_seg[cur_iter][mask_valid]
                accum_metric_segments.update_metrics(cur_logits, cur_labels)

        per_class_iou = accum_metric.per_class_iou()
        per_class_acc = accum_metric.per_class_acc()
        if use_segments:
            per_class_iou_seg = accum_metric_segments.per_class_iou()
            per_class_acc_seg = accum_metric_segments.per_class_acc()

        print()
        print("Non updated pts:", int(non_updated_pts))
        print()
        for i in range(per_class_iou.shape[0]):
            cur_class = i + 1
            if use_segments:
                print(
                    "{:15s}:   {:5.2f} [{:5.2f}]     {:5.2f} [{:5.2f}]   ({:9d})".format(
                        testds.class_names_[cur_class],
                        per_class_iou[i],
                        per_class_iou_seg[i],
                        per_class_acc[i],
                        per_class_acc_seg[i],
                        int(accum_metric.accum_gt_[cur_class]),
                    )
                )
            else:
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
        if use_segments:
            mIoU_seg = accum_metric_segments.class_mean_iou()
            mAcc_seg = accum_metric_segments.class_mean_acc()
            print(
                "IoU: {:5.2f} [{:5.2f}] | mAcc: {:5.2f} [{:5.2f}]".format(
                    mIoU, mIoU_seg, mAcc, mAcc_seg
                )
            )
        else:
            print("IoU: {:5.2f} | mAcc: {:5.2f}".format(mIoU, mAcc))

    # Save output.
    if args.save_output:
        if dataset_dict["dataset"] == "scannet20":

            print()
            print("### SAVING OUTPUT")

            # Create folders.
            colors_pred_folder = os.path.join(test_dict["save_folder"], "colors_pred")
            colors_gt_folder = os.path.join(test_dict["save_folder"], "colors_gt")
            pred_folder = os.path.join(test_dict["save_folder"], "pred")
            gt_folder = os.path.join(test_dict["save_folder"], "gt")
            if not os.path.exists(colors_pred_folder):
                os.makedirs(colors_pred_folder)
            if not os.path.exists(colors_gt_folder):
                os.makedirs(colors_gt_folder)
            if not os.path.exists(pred_folder):
                os.makedirs(pred_folder)
            if not os.path.exists(gt_folder):
                os.makedirs(gt_folder)
            if use_segments:
                colors_pred_seg_folder = os.path.join(
                    test_dict["save_folder"], "colors_pred_seg"
                )
                colors_seg_folder = os.path.join(test_dict["save_folder"], "colors_seg")
                pred_seg_folder = os.path.join(test_dict["save_folder"], "pred_seg")
                if not os.path.exists(colors_pred_seg_folder):
                    os.makedirs(colors_pred_seg_folder)
                if not os.path.exists(colors_seg_folder):
                    os.makedirs(colors_seg_folder)
                if not os.path.exists(pred_seg_folder):
                    os.makedirs(pred_seg_folder)

            # Save predictions.
            for cur_scene_iter in range(len(scene_names)):
                if cur_scene_iter % 25 == 0:
                    print(
                        "{:4d} / {:4d} {:25s}".format(
                            cur_scene_iter,
                            len(scene_names),
                            scene_names[cur_scene_iter],
                        )
                    )

                save_predictions_scannet20(
                    (colors_pred_folder, pred_folder),
                    scene_names[cur_scene_iter],
                    point_pos[cur_scene_iter],
                    np.argmax(accum_logits[cur_scene_iter], -1),
                )
                save_predictions_scannet20(
                    (colors_gt_folder, gt_folder),
                    scene_names[cur_scene_iter],
                    point_pos[cur_scene_iter],
                    point_labels[cur_scene_iter],
                )
                if use_segments:
                    save_predictions_scannet20(
                        (colors_pred_seg_folder, pred_seg_folder),
                        scene_names[cur_scene_iter],
                        point_pos[cur_scene_iter],
                        np.argmax(accum_logits_seg[cur_scene_iter], -1),
                    )
                    save_scannet20_scene_rnd_colors(
                        os.path.join(
                            colors_seg_folder, scene_names[cur_scene_iter] + ".txt"
                        ),
                        point_pos[cur_scene_iter],
                        segment_ids[cur_scene_iter],
                    )
