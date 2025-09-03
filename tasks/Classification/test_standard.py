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
from torch.utils.data import DataLoader
import point_cloud_lib as pclib
from pathlib import Path

current_milli_time = lambda: time.time() * 1000.0

MAX_NUM_THREADS = 5
GPU_ID = 0

torch.set_num_threads(MAX_NUM_THREADS)


############## DATA LOADERS
def create_data_loaders(p_ds_dict, p_batch_size, p_data_folder):

    if not p_ds_dict["test_aug_file"] == "None":
        aug_test = importlib.import_module(p_ds_dict["test_aug_file"])
    else:
        aug_test = NotImplemented

    testds = pclib.data_sets.loaders.ModelNet40DS(
        p_data_folder=p_data_folder,
        p_augmentation_cfg=aug_test.DS_AUGMENTS if not aug_test is None else None,
        p_num_pts=p_ds_dict["num_points"],
        p_split="test",
    )
    testdl = DataLoader(
        testds,
        batch_size=p_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=pclib.data_sets.loaders.ModelNet40_Collate.collate,
        num_workers=0,
    )

    num_classes = 40
    num_in_feats = 1

    return testds, testdl, num_classes, num_in_feats


############## MODEL
def create_model(p_model_dict, p_num_classes, p_num_in_feats, p_param_list):
    # Load model class.
    spec = importlib.util.spec_from_file_location(
        "models",
        "tasks/Classification/class_models.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, p_model_dict["model"])

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

    param_count = 0
    for p in list(out_models[0].parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        param_count += nn

    return out_models, param_count


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


############## TEST
def test(p_model_dict, p_models, p_data_loader, p_accum_logits, p_classes):

    with torch.no_grad():

        for cur_model in p_models:
            cur_model.eval()
        shape_iter = 0
        for cur_iter, cur_batch in enumerate(p_data_loader):

            start_batch_time = current_milli_time()
            hierarchy, features, lev_radii = create_hierarchy(
                p_model_dict,
                cur_batch[0].cuda(device=GPU_ID),
                cur_batch[2].cuda(device=GPU_ID),
                cur_batch[1].cuda(device=GPU_ID),
            )
            fake_features = torch.ones((features.shape[0], 1), device=features.device)
            for cur_model in p_models:

                pred = cur_model(hierarchy, fake_features, lev_radii).cpu().numpy()
                labels = cur_batch[3].cpu().numpy()

                for cur_shape in range(pred.shape[0]):
                    p_accum_logits[shape_iter + cur_shape, :] += pred[cur_shape, :]
                    p_classes[shape_iter + cur_shape] = labels[cur_shape]

            end_batch_time = current_milli_time()

            shape_iter += pred.shape[0]

            if cur_iter % 50 == 0:
                print(
                    "{:5d} / {:5d} ({:.1f} ms)".format(
                        cur_iter, len(p_data_loader), end_batch_time - start_batch_time
                    )
                )


def save_results(
    save_folder, accum_logits, preds, classes, class_acc_list, class_acc, acc, test_time
):
    np.savetxt(str(save_folder / "accum_logits.txt"), accum_logits)
    np.savetxt(str(save_folder / "preds.txt"), preds)
    np.savetxt(str(save_folder / "classes.txt"), classes)
    np.savetxt(str(save_folder / "class_acc_list.txt"), class_acc_list)
    np.savetxt(str(save_folder / "accum_logits.txt"), accum_logits)

    with open(str(save_folder / ("final_stats_results.txt")), "w") as text_file:
        text_file.write("Acc: {:.2f} ".format(acc * 100.0) + "\n")
        text_file.write("Class Acc: {:.2f} ".format(class_acc * 100.0) + "\n")
        text_file.write("Time {:.2f} min".format(test_time) + "\n")


############## MAIN
if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser(description="Test Classification")
    parser.add_argument(
        "--conf_file",
        default="confs/modelnet40_test_standard.yaml",
        help="Configuration file (default: confs/modelnet40_test_standard.yaml)",
    )
    # parser.add_argument(
    #     "--saved_model", default="", help="Saved model (default: model.pth)"
    # )
    parser.add_argument(
        "--data_folder",
        default="/data/lweijler/modelnet40/modelnet40_normal_resampled",
        help="Path to preprocessed data folder (default: /data/lweijler/modelnet40/modelnet40_normal_resampled)",
    )
    parser.add_argument("--gpu", type=int, default=2, help="GPU Id (default: 0)")
    args = parser.parse_args()

    # Set the gpu id.
    GPU_ID = args.gpu
    model_paths = ["path to your saved model/model_epoch_500.pth"]
    results_folder = Path("results/modelnet40/")
    for m_path in model_paths:
        exp_results_folder = results_folder / (
            "train-"
            + Path(m_path).parent.parent.name
            + "_"
            + Path(m_path).parent.name
            + "_test-SO3"
        )
        exp_results_folder.mkdir(parents=False, exist_ok=False)

        # Parse config file.
        with open(args.conf_file, "r") as f:
            conf_file = yaml.safe_load(f)
        test_dict = conf_file["Testing"]
        dataset_dict = conf_file["Dataset"]

        # Load the model parameters.
        param_list = []
        list_files = glob.glob(m_path)  # glob.glob(args.saved_model)
        for cur_file in list_files:
            dictionary = torch.load(cur_file, map_location="cpu")
            param_list.append(dictionary["params_dict"])
            model_dict = dictionary["model_dict"]
        print()

        # Create the data loaders.
        start_data_time = current_milli_time()
        testds, testdl, num_cls, num_in_feats = create_data_loaders(
            dataset_dict, test_dict["batch_size"], p_data_folder=args.data_folder
        )
        end_data_time = current_milli_time()
        print(
            "### Data Loaded ({:d} models test) {:.2f} s".format(
                len(testds), (end_data_time - start_data_time) / 1000.0
            )
        )

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

        # Create the accumulated logits.
        accum_logits = np.zeros((len(testds), num_cls), dtype=float)
        classes = np.zeros((len(testds),), dtype=int)

        # Iterate over the epochs.
        start_test_time = current_milli_time()
        for cur_epoch in range(test_dict["num_epochs"]):
            print()
            print("### EPOCH {:4d} / {:4d}".format(cur_epoch, test_dict["num_epochs"]))

            # Test.
            test(model_dict, models, testdl, accum_logits, classes)

            # Compute accuracy.
            preds = np.argmax(accum_logits, -1)
            equal_models = classes == preds
            class_acc_list = []
            for cur_class in range(num_cls):
                mask = classes == cur_class
                class_acc_list.append(np.mean(equal_models[mask]))
            accuracy = np.mean(equal_models)
            class_accuracy = np.mean(class_acc_list)

            print("Acc: {:.2f} ".format(accuracy * 100.0))
            print("Class Acc: {:.2f} ".format(class_accuracy * 100.0))

            # Increase epoch counter data set.
            testds.increase_epoch_counter()

        # Compute accuracy.
        preds = np.argmax(accum_logits, -1)
        equal_models = classes == preds
        class_acc_list = []
        for cur_class in range(num_cls):
            mask = classes == cur_class
            class_acc_list.append(np.mean(equal_models[mask]))
        accuracy = np.mean(equal_models)
        class_accuracy = np.mean(class_acc_list)

        end_test_time = current_milli_time()

        print("Acc: {:.2f} ".format(accuracy * 100.0))
        print("Class Acc: {:.2f} ".format(class_accuracy * 100.0))
        test_time = (end_test_time - start_test_time) / 60000.0
        print("Time {:.2f} min".format((end_test_time - start_test_time) / 60000.0))

        print(f"### Saving results to {exp_results_folder}")
        save_results(
            exp_results_folder,
            accum_logits,
            preds,
            classes,
            class_acc_list,
            class_accuracy,
            accuracy,
            test_time,
        )
        print("### Done")
