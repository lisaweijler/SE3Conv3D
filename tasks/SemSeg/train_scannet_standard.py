import os
import sys
import time
import argparse
import yaml
import math
import importlib
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
import point_cloud_lib as pclib
from einops import repeat

current_milli_time = lambda: time.time() * 1000.0

MAX_NUM_THREADS = 8
GPU_ID = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

torch.set_num_threads(MAX_NUM_THREADS)


############## DATA LOADERS
def create_data_loaders(p_ds_dict, p_num_batches, p_pts_per_batch):

    if "scannet" in p_ds_dict["dataset"]:

        if not p_ds_dict["train_aug_file"] == "None":
            aug_train = importlib.import_module(p_ds_dict["train_aug_file"])
        else:
            aug_train = None
        if not p_ds_dict["train_aug_color_file"] == "None":
            aug_color_train = importlib.import_module(p_ds_dict["train_aug_color_file"])
        else:
            aug_color_train = None
        if not p_ds_dict["test_aug_file"] == "None":
            aug_test = importlib.import_module(p_ds_dict["test_aug_file"])
        else:
            aug_test = None
        if not p_ds_dict["test_aug_color_file"] == "None":
            aug_color_test = importlib.import_module(p_ds_dict["test_aug_color_file"])
        else:
            aug_color_test = None

        trainds = pclib.data_sets.loaders.ScanNetDS(
            p_data_folder="/data/databases/scannet_p",
            p_dataset=p_ds_dict["dataset"],
            p_augmentation_cfg=aug_train.DS_AUGMENTS if not aug_train is None else [],
            p_augmentation_color_cfg=(
                aug_color_train.DS_AUGMENTS if not aug_color_train is None else []
            ),
            p_prob_mix3d=p_ds_dict["prob_mix3d"],
            p_split=p_ds_dict["train_split"],
        )
        train_sampler = pclib.data_sets.loaders.ScanNetMaxPtsSampler(
            p_num_batches=p_num_batches,
            p_max_points_x_batch=p_pts_per_batch,
            p_data_set=trainds,
            p_max_scene_pts=(
                p_ds_dict["train_scene_max_pts"]
                if "train_scene_max_pts" in p_ds_dict
                else 0
            ),
            p_pts_crop_ratio=(
                p_ds_dict["train_scene_crop_ratio"]
                if "train_scene_crop_ratio" in p_ds_dict
                else 1.0
            ),
        )
        traindl = DataLoader(
            trainds,
            batch_sampler=train_sampler,
            collate_fn=pclib.data_sets.loaders.ScanNet_Collate.collate,
            num_workers=3,
        )

        testds = pclib.data_sets.loaders.ScanNetDS(
            p_data_folder="/data/databases/scannet_p",
            p_dataset=p_ds_dict["dataset"],
            p_augmentation_cfg=aug_test.DS_AUGMENTS if not aug_test is None else [],
            p_augmentation_color_cfg=(
                aug_color_test.DS_AUGMENTS if not aug_color_test is None else []
            ),
            p_prob_mix3d=0.0,
            p_split=p_ds_dict["test_split"],
        )
        testdl = DataLoader(
            testds,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=pclib.data_sets.loaders.ScanNet_Collate.collate,
            num_workers=1,
        )

        num_classes = 21
        num_in_feats = 3
        mask_classes = [0]

    return trainds, traindl, testds, testdl, num_classes, num_in_feats, mask_classes


############## MODEL
def create_model(p_model_dict, p_num_classes, p_num_in_feats):
    spec = importlib.util.spec_from_file_location(
        "models",
        "/caa/Homes01/lweijler/phd/point_clouds/point_clouds_nn/Tasks_Rot_Equiv/SemSeg/seg_models.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, p_model_dict["model"])
    model = model_class(
        p_num_in_feats=p_num_in_feats,
        p_num_out_classes=p_num_classes,
        p_max_path_drop=p_model_dict["max_drop_path"],
    )

    model.cuda(device=GPU_ID)

    param_count = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        param_count += nn

    return model, param_count


############## CREATE HIERARCHY
def create_hierarchy(
    p_model_dict, p_pts, p_batch_ids, p_features, p_init_subsample=False, p_labels=None
):
    with torch.no_grad():
        pc = pclib.pc.Pointcloud(p_pts, p_batch_ids)

        samp = pclib.pc.GridSubSample(pc, p_model_dict["init_subsample"])
        new_pts = samp.__subsample_tensor__(pc.pts_, "avg")
        new_batch_ids = samp.__subsample_tensor__(pc.batch_ids_, "max")
        new_features = samp.__subsample_tensor__(p_features, "avg")
        new_pc = pclib.pc.Pointcloud(new_pts, new_batch_ids)

        new_features = new_features[:, 3:]

        hierarchy = pclib.pc.PointHierarchy(
            new_pc,
            len(p_model_dict["grid_subsamples"]),
            "grid_avg",
            grid_radii=p_model_dict["grid_subsamples"],
        )

        levels_radii = [p_model_dict["init_subsample"]] + p_model_dict[
            "grid_subsamples"
        ]

        if p_init_subsample:
            if "output_subsample" in p_model_dict:
                out_subsample_radius = p_model_dict["output_subsample"]
                samp = pclib.pc.GridSubSample(
                    pc, out_subsample_radius, p_rnd_sample=True
                )
                new_init_pts = samp.__subsample_tensor__(pc.pts_, "avg")
                new_init_batch_ids = samp.__subsample_tensor__(pc.batch_ids_, "max")
                new_init_labels = samp.__subsample_tensor__(p_labels, "max")
                new_init_pc = pclib.pc.Pointcloud(new_init_pts, new_init_batch_ids)
            else:
                new_init_pc = pc
                new_init_labels = p_labels

            return new_init_pc, new_init_labels, hierarchy, new_features, levels_radii
        else:
            return pc, p_labels, hierarchy, new_features, levels_radii


############## MASK VALID POINTS
def mask_valid_points(p_labels, p_mask_cls):
    with torch.no_grad():
        cur_mask = torch.ones_like(p_labels).to(torch.bool)
        for cur_class in p_mask_cls:
            aux_mask = p_labels != cur_class
            cur_mask = torch.logical_and(cur_mask, aux_mask)
        return cur_mask


############## PRE-PROCESS
def pre_process(p_model_dict, p_model, p_data_loader, p_mix_prec):
    with torch.no_grad():

        p_model.eval()
        p_model.start_pre_process()
        for cur_iter, cur_batch in enumerate(p_data_loader):
            start_batch_time = current_milli_time()
            labels = cur_batch[3].cuda(device=GPU_ID)
            init_pc, _, hierarchy, features, lev_radii = create_hierarchy(
                p_model_dict,
                cur_batch[0].cuda(device=GPU_ID),
                cur_batch[2].cuda(device=GPU_ID),
                cur_batch[1].cuda(device=GPU_ID),
                True,
                labels,
            )
            mid_batch_time = current_milli_time()

            ##################
            # for cur_batch_id in range(torch.amax(cur_batch[2]).item()+1):
            #    cur_mask = cur_batch[2] == cur_batch_id
            #    np.savetxt("scene_"+str(cur_batch_id)+".txt", cur_batch[0][cur_mask].detach().cpu().numpy())
            # assert(1 == 0)
            ##################

            if p_mix_prec:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    p_model(hierarchy, features, lev_radii, init_pc)
            else:
                p_model(hierarchy, features, lev_radii, init_pc)
            end_batch_time = current_milli_time()
            if cur_iter % 50 == 0:
                print(
                    "Pre-process {:5d} / {:5d} ({:.1f} ms, {:.1f} ms)".format(
                        cur_iter,
                        len(p_data_loader),
                        mid_batch_time - start_batch_time,
                        end_batch_time - mid_batch_time,
                    )
                )
        p_model.end_pre_process()


############## TRAIN
def train(
    p_train_dict,
    p_model_dict,
    p_model,
    p_data_loader,
    p_loss_fn,
    p_optim,
    p_scheduler,
    p_num_cls,
    p_mask_cls,
    p_mix_prec,
    p_grad_scaler,
):

    accum_loss = 0.0
    accum_metric = pclib.metrics.SemSegMetrics(p_num_cls, p_mask_cls)
    p_model.train()
    for cur_iter, cur_batch in enumerate(p_data_loader):

        start_batch_time = current_milli_time()
        labels = cur_batch[3].cuda(device=GPU_ID)
        init_pc, labels, hierarchy, features, lev_radii = create_hierarchy(
            p_model_dict,
            cur_batch[0].cuda(device=GPU_ID),
            cur_batch[2].cuda(device=GPU_ID),
            cur_batch[1].cuda(device=GPU_ID),
            True,
            labels,
        )

        cur_mask = mask_valid_points(labels, p_mask_cls)
        out_pc = pclib.pc.Pointcloud(
            init_pc.pts_[cur_mask], init_pc.batch_ids_[cur_mask]
        )
        labels = labels[cur_mask]

        if p_mix_prec:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = p_model(hierarchy, features, lev_radii, out_pc)
                loss = p_loss_fn(pred, labels)
            p_grad_scaler.scale(loss).backward()
            if p_train_dict["clip_grads"] > 0.0:
                p_grad_scaler.unscale_(p_optim)
                torch.nn.utils.clip_grad_norm_(
                    p_model.parameters(), p_train_dict["clip_grads"]
                )
            p_grad_scaler.step(p_optim)
            p_grad_scaler.update()
        else:

            pred = p_model(hierarchy, features, lev_radii, out_pc)
            loss = p_loss_fn(pred, labels)
            loss.backward()
            if p_train_dict["clip_grads"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    p_model.parameters(), p_train_dict["clip_grads"]
                )
            p_optim.step()

        p_optim.zero_grad()

        p_scheduler.step()

        accum_metric.update_metrics(
            pred.detach().cpu().numpy(), labels.detach().cpu().numpy()
        )

        accum_loss += (loss.item() - accum_loss) / (cur_iter + 1)

        end_batch_time = current_milli_time()

        if cur_iter % 25 == 0:
            print(
                "{:5d} / {:5d} | Loss: {:.4f} | mIoU: {:.2f} | mAcc: {:.2f} | ({:.1f} ms)".format(
                    cur_iter,
                    len(p_data_loader),
                    accum_loss,
                    accum_metric.class_mean_iou(),
                    accum_metric.class_mean_acc(),
                    end_batch_time - start_batch_time,
                )
            )

    return accum_metric.class_mean_iou(), accum_metric.class_mean_acc(), accum_loss


############## VAL
def validation(
    p_model_dict, p_model, p_data_loader, p_loss_fn, p_num_cls, p_mask_cls, p_mix_prec
):

    with torch.no_grad():

        accum_loss = 0.0
        accum_metric = pclib.metrics.SemSegMetrics(p_num_cls, p_mask_cls)
        p_model.eval()
        for cur_iter, cur_batch in enumerate(p_data_loader):

            start_batch_time = current_milli_time()
            labels = cur_batch[3].cuda(device=GPU_ID)
            init_pc, labels, hierarchy, features, lev_radii = create_hierarchy(
                p_model_dict,
                cur_batch[0].cuda(device=GPU_ID),
                cur_batch[2].cuda(device=GPU_ID),
                cur_batch[1].cuda(device=GPU_ID),
                False,
                labels,
            )

            cur_mask = mask_valid_points(labels, p_mask_cls)
            out_pc = pclib.pc.Pointcloud(
                init_pc.pts_[cur_mask], init_pc.batch_ids_[cur_mask]
            )
            labels = labels[cur_mask]

            if p_mix_prec:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = p_model(hierarchy, features, lev_radii, out_pc)
                    loss = p_loss_fn(pred, labels)
            else:
                pred = p_model(hierarchy, features, lev_radii, out_pc)
                loss = p_loss_fn(pred, labels)

            accum_loss += (loss.item() - accum_loss) / (cur_iter + 1)

            accum_metric.update_metrics(pred.cpu().numpy(), labels.cpu().numpy())

            end_batch_time = current_milli_time()

            if cur_iter % 100 == 0:
                print(
                    "{:5d} / {:5d} | Loss: {:.4f} | mIoU: {:.2f} | mAcc: {:.2f} | ({:.1f} ms)".format(
                        cur_iter,
                        len(p_data_loader),
                        accum_loss,
                        accum_metric.class_mean_iou(),
                        accum_metric.class_mean_acc(),
                        end_batch_time - start_batch_time,
                    )
                )

    print()
    per_class_iou = accum_metric.per_class_iou()
    per_class_acc = accum_metric.per_class_acc()
    for i in range(per_class_iou.shape[0]):
        cur_class = i + 1
        print(
            "{:15s}: {:5.2f} {:5.2f} ({:9d})".format(
                p_data_loader.dataset.class_names_[cur_class],
                per_class_iou[i],
                per_class_acc[i],
                int(accum_metric.accum_gt_[cur_class]),
            )
        )
    print()
    return accum_metric.class_mean_iou(), accum_metric.class_mean_acc(), accum_loss


############## SAVE MODEL
def save_checkpoint(
    p_file_name,
    p_train_dict,
    p_dataset_dict,
    p_model_dict,
    p_params_dict,
    p_optimizer_dict,
    p_scheduler_dict,
    p_best_mIoU,
    epoch,
):

    save_dict = {}
    save_dict["train_dict"] = p_train_dict
    save_dict["dataset_dict"] = p_dataset_dict
    save_dict["model_dict"] = p_model_dict
    save_dict["params_dict"] = p_params_dict
    save_dict["optimizer_dict"] = p_optimizer_dict
    save_dict["scheduler_dict"] = p_scheduler_dict
    save_dict["best_mIoU"] = p_best_mIoU
    save_dict["epoch"] = epoch
    torch.save(save_dict, p_file_name)


############## MAIN
if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser(description="Train Sematic segmentation")
    parser.add_argument(
        "--conf_file",
        default="/caa/Homes01/lweijler/phd/point_clouds/point_clouds_nn/Tasks_Rot_Equiv/SemSeg/confs/cvpr24/eccv_resubmission/scannet20_standard_I.yaml",
        help="Configuration file (default: confs/scannet20.yaml)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU Id (default: 0)")
    parser.add_argument(
        "--mix_prec",
        dest="mix_prec",
        action="store_true",
        help="Train with mixed precission (default: False",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args = parser.parse_args()

    # Set the gpu id.
    GPU_ID = args.gpu

    # Parse config file.
    with open(args.conf_file, "r") as f:
        conf_file = yaml.safe_load(f)
    train_dict = conf_file["Training"]
    dataset_dict = conf_file["Dataset"]
    model_dict = conf_file["Model"]
    print()

    # Init WandB
    wandb.init(
        entity="cvl-myeflow",
        project="3DV25",
        group="semseg_" + dataset_dict["dataset"],
        name=train_dict["log_folder"].split("/")[-1],
        config={**train_dict, **dataset_dict, **model_dict},
    )

    # wandb.init(
    #     entity="your_entity",  # replace with your WandB entity
    #     project="SE3Conv3D",
    #     group="semseg_" + dataset_dict["dataset"],
    #     name=train_dict["log_folder"].split("/")[-1],
    #     config={**train_dict, **dataset_dict, **model_dict},
    # )
    print()

    # Create the log folder.
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(train_dict["log_folder"]):
        os.makedirs(train_dict["log_folder"])
    os.makedirs(train_dict["log_folder"] + "/" + timestr)

    with open(
        train_dict["log_folder"] + "/" + timestr + "/used_conf.yaml", "w"
    ) as file:
        yaml.dump(conf_file, file)

    # Create the data loaders.
    start_data_time = current_milli_time()
    trainds, traindl, testds, testdl, num_cls, num_in_feats, mask_cls = (
        create_data_loaders(
            p_ds_dict=dataset_dict,
            p_num_batches=train_dict["num_batches"],
            p_pts_per_batch=train_dict["pts_per_batch"],
        )
    )
    end_data_time = current_milli_time()
    print(
        "### Data Loaded ({:d} models train) ({:d} models test) {:.2f} s".format(
            len(trainds), len(testds), (end_data_time - start_data_time) / 1000.0
        )
    )

    # Iterate over the epochs.
    save_counter = 0
    best_mIoU = 0.0
    start_epoch = 0
    params = None
    # resume model training - careful no check if model dict traindict, data dict are same
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        best_mIoU = checkpoint["best_mIoU"]
        params = checkpoint["params_dict"]

    # Create the model.
    start_model_time = current_milli_time()
    model, param_count = create_model(model_dict, num_cls, num_in_feats)
    end_model_time = current_milli_time()
    print(
        "### Model {:s} Created ({:d} params) {:.2f} ms".format(
            model_dict["model"], param_count, end_model_time - start_model_time
        )
    )

    # Create optimizer.
    print("### Optimizer")
    param_groups = [
        {
            "params": model.parameters(),
            "weight_decay": train_dict["weight_decay"],
            "lr": train_dict["max_lr"] / train_dict["div_factor"],
        }
    ]
    optimizer = torch.optim.AdamW(param_groups)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_dict["max_lr"],
        steps_per_epoch=len(traindl),
        epochs=train_dict["num_epochs"] + 1,
        div_factor=train_dict["div_factor"],
        final_div_factor=train_dict["final_div_factor"],
        pct_start=train_dict["pct_start"],
    )

    if args.resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_dict"])

    # Create the loss function.
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="mean", label_smoothing=train_dict["label_smoothing"]
    )

    # Preprocess data set.
    print()
    print("### Pre-Process")
    start_preproc_time = current_milli_time()
    pre_process(model_dict, model, traindl, args.mix_prec)
    end_preproc_time = current_milli_time()
    print("Time {:.2} min".format((end_preproc_time - start_preproc_time) / 60000.0))

    # Iterate over the epochs.

    if args.mix_prec:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None
    for cur_epoch in range(train_dict["num_epochs"]):
        print()
        print("### EPOCH {:4d} / {:4d}".format(cur_epoch, train_dict["num_epochs"]))

        # Train.
        print("# TRAIN")

        start_epoch_time = current_milli_time()
        train_iou, train_acc, train_loss = train(
            train_dict,
            model_dict,
            model,
            traindl,
            loss_fn,
            optimizer,
            lr_scheduler,
            num_cls,
            mask_cls,
            args.mix_prec,
            grad_scaler,
        )
        end_epoch_time = current_milli_time()

        print(
            "mIoU: {:.2f} | mAcc: {:.2f} | Loss: {:.4f}".format(
                train_iou, train_acc, train_loss
            )
        )
        print("Time {:.2f} min".format((end_epoch_time - start_epoch_time) / 60000.0))

        # Validation.
        if cur_epoch % train_dict["val_freq"] == 0:
            print("# VAL")

            start_epoch_time = current_milli_time()
            val_iou, val_acc, val_loss = validation(
                model_dict, model, testdl, loss_fn, num_cls, mask_cls, args.mix_prec
            )
            end_epoch_time = current_milli_time()

            print(
                "mIoU: {:.2f} [{:.2f}] | mAcc: {:.2f} | Loss: {:.4f}".format(
                    val_iou, best_mIoU, val_acc, val_loss
                )
            )
            print(
                "Time {:.2f} min".format((end_epoch_time - start_epoch_time) / 60000.0)
            )

            wandb.log(
                {
                    "train_iou": train_iou,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "val_iou": val_iou,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )

            if val_iou > best_mIoU:
                best_mIoU = val_iou
                save_checkpoint(
                    p_file_name=train_dict["log_folder"] + "/" + timestr + "/best.pth",
                    p_train_dict=train_dict,
                    p_dataset_dict=dataset_dict,
                    p_model_dict=model_dict,
                    p_params_dict=model.state_dict(),
                    p_optimizer_dict=optimizer.state_dict(),
                    p_scheduler_dict=lr_scheduler.state_dict(),
                    p_best_mIoU=best_mIoU,
                    epoch=cur_epoch,
                )
        else:
            wandb.log(
                {
                    "train_iou": train_iou,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )

        # Save models.
        if 0 == cur_epoch or (
            0 == cur_epoch % train_dict["save_models_frequency"]
            or cur_epoch == train_dict["num_epochs"] - 1
        ):
            save_checkpoint(
                p_file_name=train_dict["log_folder"]
                + "/"
                + timestr
                + "/model_epoch_"
                + str(cur_epoch)
                + ".pth",
                p_train_dict=train_dict,
                p_dataset_dict=dataset_dict,
                p_model_dict=model_dict,
                p_params_dict=model.state_dict(),
                p_optimizer_dict=optimizer.state_dict(),
                p_scheduler_dict=lr_scheduler.state_dict(),
                p_best_mIoU=best_mIoU,
                epoch=cur_epoch,
            )
