import os
import time
import argparse
import yaml
import importlib
import torch
import wandb
from torch.utils.data import DataLoader
import point_cloud_lib as pclib

current_milli_time = lambda: time.time() * 1000.0

MAX_NUM_THREADS = 3
GPU_ID = 0

torch.set_num_threads(MAX_NUM_THREADS)


############## DATA LOADERS
def create_data_loaders(p_ds_dict, p_train_batch, p_test_batch, p_data_folder):

    if not p_ds_dict["train_aug_file"] == "None":
        aug_train = importlib.import_module(p_ds_dict["train_aug_file"])
    else:
        aug_train = None
    if not p_ds_dict["test_aug_file"] == "None":
        aug_test = importlib.import_module(p_ds_dict["test_aug_file"])
    else:
        aug_train = None

    trainds = pclib.data_sets.loaders.ModelNet40DS(
        p_data_folder=p_data_folder,
        p_augmentation_cfg=aug_train.DS_AUGMENTS if not aug_train is None else [],
        p_num_pts=p_ds_dict["num_points"],
        p_split="train",
    )
    traindl = DataLoader(
        trainds,
        batch_size=p_train_batch,
        shuffle=True,
        drop_last=True,
        collate_fn=pclib.data_sets.loaders.ModelNet40_Collate.collate,
        num_workers=1,
    )

    testds = pclib.data_sets.loaders.ModelNet40DS(
        p_data_folder=p_data_folder,
        p_augmentation_cfg=aug_test.DS_AUGMENTS if not aug_test is None else [],
        p_num_pts=p_ds_dict["num_points"],
        p_split="test",
    )
    testdl = DataLoader(
        testds,
        batch_size=p_test_batch,
        shuffle=False,
        drop_last=False,
        collate_fn=pclib.data_sets.loaders.ModelNet40_Collate.collate,
        num_workers=1,
    )

    num_classes = 40
    num_in_feats = 1

    return trainds, traindl, testds, testdl, num_classes, num_in_feats


############## MODEL


def create_model(p_model_dict, p_num_classes, p_num_in_feats):
    # Load model class.
    spec = importlib.util.spec_from_file_location(
        "models",
        "tasks/Classification/class_models.py",
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
def create_hierarchy(p_model_dict, p_pts, p_batch_ids, p_features):
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
        return hierarchy, new_features, levels_radii


############## PRE-PROCESS
def pre_process(p_model_dict, p_model, p_data_loader):
    with torch.no_grad():

        p_model.eval()
        p_model.start_pre_process()
        for cur_iter, cur_batch in enumerate(p_data_loader):
            start_batch_time = current_milli_time()
            hierarchy, features, lev_radii = create_hierarchy(
                p_model_dict,
                cur_batch[0].cuda(device=GPU_ID),
                cur_batch[2].cuda(device=GPU_ID),
                cur_batch[1].cuda(device=GPU_ID),
            )
            fake_features = torch.ones((features.shape[0], 1), device=features.device)
            p_model(hierarchy, fake_features, lev_radii)
            end_batch_time = current_milli_time()
            if cur_iter % 50 == 0:
                print(
                    "Pre-process {:5d} / {:5d} ({:.1f} ms)".format(
                        cur_iter, len(p_data_loader), end_batch_time - start_batch_time
                    )
                )
        p_model.end_pre_process()


############## TRAIN
def train(
    p_train_dict, p_model_dict, p_model, p_data_loader, p_loss_fn, p_optim, p_scheduler
):

    accum_loss = 0.0
    accum_acc = 0.0
    p_model.train()
    for cur_iter, cur_batch in enumerate(p_data_loader):

        start_batch_time = current_milli_time()
        hierarchy, features, lev_radii = create_hierarchy(
            p_model_dict,
            cur_batch[0].cuda(device=GPU_ID),
            cur_batch[2].cuda(device=GPU_ID),
            cur_batch[1].cuda(device=GPU_ID),
        )

        fake_features = torch.ones((features.shape[0], 1), device=features.device)
        pred = p_model(hierarchy, fake_features, lev_radii)
        labels = cur_batch[3].cuda(device=GPU_ID)

        loss = p_loss_fn(pred, labels)
        loss.backward()

        cur_acc = torch.mean((torch.argmax(pred, -1) == labels).to(torch.float32))

        accum_loss += (loss.item() - accum_loss) / (cur_iter + 1)
        accum_acc += (cur_acc.item() - accum_acc) / (cur_iter + 1)

        torch.nn.utils.clip_grad_norm_(p_model.parameters(), p_train_dict["clip_grads"])

        p_optim.step()
        p_optim.zero_grad()

        p_scheduler.step()

        end_batch_time = current_milli_time()

        if cur_iter % 50 == 0:
            print(
                "{:5d} / {:5d} | Loss: {:.4f} | Acc: {:.2f} | ({:.1f} ms)".format(
                    cur_iter,
                    len(p_data_loader),
                    accum_loss,
                    accum_acc * 100.0,
                    end_batch_time - start_batch_time,
                )
            )

    return accum_acc, accum_loss


############## VAL
def validation(p_model_dict, p_model, p_data_loader, p_loss_fn):

    with torch.no_grad():

        accum_loss = 0.0
        accum_acc = 0.0
        p_model.eval()
        for cur_iter, cur_batch in enumerate(p_data_loader):

            start_batch_time = current_milli_time()
            hierarchy, features, lev_radii = create_hierarchy(
                p_model_dict,
                cur_batch[0].cuda(device=GPU_ID),
                cur_batch[2].cuda(device=GPU_ID),
                cur_batch[1].cuda(device=GPU_ID),
            )
            fake_features = torch.ones((features.shape[0], 1), device=features.device)
            pred = p_model(hierarchy, fake_features, lev_radii)
            labels = cur_batch[3].cuda(device=GPU_ID)

            loss = p_loss_fn(pred, labels)

            cur_acc = torch.mean((torch.argmax(pred, -1) == labels).to(torch.float32))

            accum_loss += (loss.item() - accum_loss) / (cur_iter + 1)
            accum_acc += (cur_acc.item() - accum_acc) / (cur_iter + 1)

            end_batch_time = current_milli_time()

            if cur_iter % 50 == 0:
                print(
                    "{:5d} / {:5d} | Loss: {:.4f} | Acc: {:.2f} | ({:.1f} ms)".format(
                        cur_iter,
                        len(p_data_loader),
                        accum_loss,
                        accum_acc * 100.0,
                        end_batch_time - start_batch_time,
                    )
                )

    return accum_acc, accum_loss


############## SAVE MODEL
def save_checkpoint(
    p_file_name,
    p_train_dict,
    p_dataset_dict,
    p_model_dict,
    p_params_dict,
    p_optimizer_dict,
    p_scheduler_dict,
    p_best_acc,
    epoch,
):

    save_dict = {}
    save_dict["train_dict"] = p_train_dict
    save_dict["dataset_dict"] = p_dataset_dict
    save_dict["model_dict"] = p_model_dict
    save_dict["params_dict"] = p_params_dict
    save_dict["optimizer_dict"] = p_optimizer_dict
    save_dict["scheduler_dict"] = p_scheduler_dict
    save_dict["best_acc"] = p_best_acc
    save_dict["epoch"] = epoch
    torch.save(save_dict, p_file_name)


############## MAIN
if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser(description="Train Classification")
    parser.add_argument(
        "--conf_file",
        default="confs/modelnet40_standard.yaml",
        help="Configuration file (default: confs/modelnet40_standard.yaml)",
    )
    parser.add_argument(
        "--data_folder",
        default="/data/lweijler/modelnet40/modelnet40_normal_resampled",
        help="Path to preprocessed data folder (default: /data/lweijler/modelnet40/modelnet40_normal_resampled)",
    )
    parser.add_argument("--gpu", type=int, default=1, help="GPU Id (default: 0)")
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
        group="classification_" + dataset_dict["dataset"],
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
    trainds, traindl, testds, testdl, num_cls, num_in_feats = create_data_loaders(
        p_ds_dict=dataset_dict,
        p_train_batch=train_dict["batch_size"],
        p_test_batch=train_dict["batch_size"],
        p_data_folder=args.data_folder,
    )
    end_data_time = current_milli_time()
    print(
        "### Data Loaded ({:d} models train) ({:d} models test) {:.2f} s".format(
            len(trainds), len(testds), (end_data_time - start_data_time) / 1000.0
        )
    )

    # Iterate over the epochs.
    save_counter = 0
    best_acc = 0.0
    start_epoch = 0
    params = None
    # resume model training - careful no check if model dict traindict, data dict are same
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
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
    save_counter = 0
    best_acc = 0.0
    for cur_epoch in range(train_dict["num_epochs"]):
        if start_epoch != 0 and cur_epoch <= start_epoch:
            continue
        print()
        print("### EPOCH {:4d} / {:4d}".format(cur_epoch, train_dict["num_epochs"]))

        # Train.
        print("# TRAIN")

        start_epoch_time = current_milli_time()
        train_acc, train_loss = train(
            train_dict, model_dict, model, traindl, loss_fn, optimizer, lr_scheduler
        )
        end_epoch_time = current_milli_time()

        print("Acc: {:.2f} | Loss: {:.4f}".format(train_acc * 100.0, train_loss))
        print("Time {:.2f} min".format((end_epoch_time - start_epoch_time) / 60000.0))

        if cur_epoch % train_dict["val_freq"] == 0:
            # Validation.
            print("# VAL")

            start_epoch_time = current_milli_time()
            val_acc, val_loss = validation(model_dict, model, testdl, loss_fn)
            end_epoch_time = current_milli_time()

            print(
                "Acc: {:.2f} [{:.2f}] | Loss: {:.4f}".format(
                    val_acc * 100.0, best_acc * 100.0, val_loss
                )
            )
            print(
                "Time {:.2f} min".format((end_epoch_time - start_epoch_time) / 60000.0)
            )

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    p_file_name=train_dict["log_folder"] + "/" + timestr + "/best.pth",
                    p_train_dict=train_dict,
                    p_dataset_dict=dataset_dict,
                    p_model_dict=model_dict,
                    p_params_dict=model.state_dict(),
                    p_optimizer_dict=optimizer.state_dict(),
                    p_scheduler_dict=lr_scheduler.state_dict(),
                    p_best_acc=best_acc,
                    epoch=cur_epoch,
                )

            wandb.log(
                {
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )

        else:
            wandb.log(
                {
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
                p_best_acc=best_acc,
                epoch=cur_epoch,
            )
