import os
import time
import argparse
import yaml
import importlib
import torch
import wandb
from torch.utils.data import DataLoader
import point_cloud_lib as pclib
from einops import repeat
from matplotlib.pyplot import get_cmap

current_milli_time = lambda: time.time() * 1000.0

MAX_NUM_THREADS = 8
GPU_ID = 0

GT_PLOTTED = False

torch.set_num_threads(MAX_NUM_THREADS)

cm = get_cmap("tab20")


############## DATA LOADERS
def create_data_loaders(p_ds_dict, p_batch_size, p_data_folder):

    if "dfaust" in p_ds_dict["dataset"]:

        if not p_ds_dict["train_aug_file"] == "None":
            aug_train = importlib.import_module(p_ds_dict["train_aug_file"])
        else:
            aug_train = None
        if not p_ds_dict["test_aug_file"] == "None":
            aug_test = importlib.import_module(p_ds_dict["test_aug_file"])
        else:
            aug_test = None

        trainds = pclib.data_sets.loaders.DFaustDS(
            p_data_folder=p_data_folder,
            p_augmentation_cfg=aug_train.DS_AUGMENTS if not aug_train is None else [],
            p_num_pts=p_ds_dict["num_points"],  # not used yet
            p_split=p_ds_dict["train_split"],
        )

        traindl = DataLoader(
            trainds,
            batch_size=p_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=pclib.data_sets.loaders.DFaust_Collate.collate,
            num_workers=3,
        )

        testds = pclib.data_sets.loaders.DFaustDS(
            p_data_folder=p_data_folder,  # "/data/lweijler/SE3Conv3D/dfaust",
            p_augmentation_cfg=aug_test.DS_AUGMENTS if not aug_test is None else [],
            p_num_pts=p_ds_dict["num_points"],
            p_split=p_ds_dict["test_split"],
        )

        testdl = DataLoader(
            testds,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=pclib.data_sets.loaders.DFaust_Collate.collate,
            num_workers=3,
        )

        num_classes = 20
        num_in_feats = 1
        mask_cls = []  # [10, 11, 22]

        return trainds, traindl, testds, testdl, num_classes, num_in_feats, mask_cls


############## MODEL
def create_model(p_model_dict, p_num_classes, p_num_in_feats):
    spec = importlib.util.spec_from_file_location(
        "models",
        "/caa/Homes01/lweijler/phd/point_clouds/published_repos/SE3Conv3D/tasks/SemSeg/seg_models.py",
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
def pre_process(p_model_dict, p_model, p_data_loader):
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

        pred = p_model(hierarchy, features, lev_radii, out_pc)
        loss = p_loss_fn(pred, labels) / p_train_dict["accum_grads"]
        loss.backward()
        if cur_iter % p_train_dict["accum_grads"] == 0:
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
def validation(p_model_dict, p_model, p_data_loader, p_loss_fn, p_num_cls, p_mask_cls):

    with torch.no_grad():
        log_images = {"val_point_cloud_pred": [], "val_point_cloud_gt": []}
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

            pred = p_model(hierarchy, features, lev_radii, out_pc)
            ##log poin cloud in wandb
            if cur_iter == 0:
                batch_id_mask = out_pc.batch_ids_ == 0
                log_pc = out_pc.pts_[batch_id_mask]
                log_pred = torch.argmax(pred[batch_id_mask], -1)

                log_pred_c = torch.stack(
                    [torch.tensor(cm.colors[i]) * 255 for i in log_pred]
                )

                point_cloud_pred = torch.cat((log_pc.cpu(), log_pred_c), dim=1).numpy()
                log_images["val_point_cloud_pred"].append(
                    wandb.Object3D(point_cloud_pred)
                )
                # wandb.log({"val_point_cloud_pred": wandb.Object3D(point_cloud_pred)})

                global GT_PLOTTED
                if not GT_PLOTTED:
                    batch_id_mask = out_pc.batch_ids_ == 0
                    log_labels = labels[batch_id_mask]
                    log_labels_c = torch.stack(
                        [torch.tensor(cm.colors[i]) * 255 for i in log_labels]
                    )
                    point_cloud_gt = torch.cat(
                        (log_pc.cpu(), log_labels_c), dim=1
                    ).numpy()
                    log_images["val_point_cloud_gt"].append(
                        wandb.Object3D(point_cloud_gt)
                    )
                    # wandb.log({"val_point_cloud_gt": wandb.Object3D(point_cloud_gt)})
                    GT_PLOTTED = True
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
        cur_class = i
        print(
            "{:15s}: {:5.2f} {:5.2f} ({:9d})".format(
                p_data_loader.dataset.class_names_[cur_class],
                per_class_iou[i],
                per_class_acc[i],
                int(accum_metric.accum_gt_[cur_class]),
            )
        )
    print()
    return (
        accum_metric.class_mean_iou(),
        accum_metric.class_mean_acc(),
        accum_loss,
        log_images,
    )


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
        default="confs/dfaust/dfaust_I_standard.yaml",
        help="Configuration file (default: confs/dfaust/dfaust_I_rot_pca_2F.yaml)",
    )
    parser.add_argument(
        "--data_folder",
        default="/data/lweijler/SE3Conv3D/dfaust",
        help="Path to preprocessed data folder (default: /data/lweijler/SE3Conv3D/dfaust)",
    )
    parser.add_argument("--gpu", type=int, default=5, help="GPU Id (default: 0)")

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
        entity="your_entity",  # replace with your WandB entity
        project="SE3Conv3D",
        group="semseg_" + dataset_dict["dataset"],
        name=train_dict["log_folder"].split("/")[-1],
        config={**train_dict, **dataset_dict, **model_dict},
    )
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
            p_batch_size=train_dict["batch_size"],
            p_data_folder=args.data_folder,
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
    pre_process(model_dict, model, traindl)
    end_preproc_time = current_milli_time()
    print("Time {:.2} min".format((end_preproc_time - start_preproc_time) / 60000.0))

    # Iterate over the epochs.

    if not "accum_grads" in train_dict:
        train_dict["accum_grads"] = 1
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
            val_iou, val_acc, val_loss, log_images = validation(
                model_dict, model, testdl, loss_fn, num_cls, mask_cls
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
            log_dict = {
                "train_iou": train_iou,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "lr": lr_scheduler.get_last_lr()[0],
                "val_iou": val_iou,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            log_dict.update(log_images)
            wandb.log(log_dict)

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
