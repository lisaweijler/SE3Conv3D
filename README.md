<span align="center">
<h1> Efficient Continuous Group Convolutions for Local SE(3) Equivariance in 3D Point Clouds</h1>
![OpenNeRF Teaser]([https://opennerf.github.io/static/images/teaser.png](https://github.com/lisaweijler/se3conv3d-projectpage/blob/main/static/images/teaser_horizontal.jpg))
<a href="https://lisaweijler.github.io/">Lisa Weijler</a>,
<a href="https://phermosilla.github.io/">Pedro Hermosilla</a>

<h3>3DV 2025</h3>

<a href="https://arxiv.org/pdf/2502.07505">Paper</a> |
<a href="https://lisaweijler.github.io/se3conv3d-projectpage/">Project Page</a> 

</span>


## 🛠️ Installation
The code was tested using **Python 3.10.12**. 

We recommend using conda to install the right CUDA dependencies.
```
conda create -n se3conv3d python=3.10 
conda activate se3conv3d
conda install -c conda-forge cudatoolkit-dev=11.7
conda install pytorch=2.0.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch-scatter==2.1.1+pt20cu117 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-cluster==1.6.1+pt20cu117 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install wheel
```
Then navigate into the point_cloud_lib folder to build and install binaries of cuda extensions:
```bash
cd point_cloud_lib
python setup.py bdist_wheel
pip install dist/point_cloud_lib-0.0.0-cp310-cp310-linux_x86_64.whl
```
Finally, install the  remaining packages used:

```bash
conda install einops=0.8.1 wandb=0.20.1 joblib=1.5.1 matplotlib=3.9.1 trimesh=4.6.13 tqdm=4.67.1 webdataset=1.0.2 h5py=3.9.0
pip install smplx=0.1.28
```

## 🚀 Experiments
A description of how to replicate the experiments from the paper is given below. Note that the `point_cloud_lib` contains all the code for the Feature Pyramid Network (FPN), and that the relevant scripts for making it rotation equivariant are
- `point_cloud_lib/point_cloud_lib/layers/PNEConvLayerRotEquiv.py`: containing the rot. equivariant convolution layer,
- `point_cloud_lib/point_cloud_lib/pc/PointcloudRotEquiv.py`: augmenting the standard pointcloud class with SO(3) samples (reference frames),
- `point_cloud_lib/point_cloud_lib/pc/PointHierarchyRotEquiv.py`: creating the point cloud hierarchy using the rot. equivariant point cloud class.

We are using weights\&biases; if you don't want to or cannot use it, simply remove all wandb-related stuff in the train files. If you use it, please specify your parameters in the wandb.init(...) function (in the train files).

### Human Body Parts Segmentation
Train and test scripts, as well as configs, are saved under `tasks/SemSeg` and denoted with `dfaust`.
#### Dataset
For training and testing, we use two subsets of the [AMASS meta-dataset](https://amass.is.tue.mpg.de/index.html), provided by the [ArtEq](https://github.com/HavenFeng/ArtEq/tree/726287fcba0b8a1306b4370ec91661e236eb1909) repository. The data can be downloaded from [here](https://download.is.tue.mpg.de/download.php?domain=arteq&sfile=data.zip&resume=1); note that you first have to create an account at https://arteq.is.tue.mpg.de/.

After unzipping, you should have the following folder structure
```
data/
├── DFaust_67_val
├── MPI_Limits
├── papermodel
├── smpl_model
└── DFaust_67_train.pth.tar
```


For preprocessing the dataset, you can use the `preprocess_data/preprocess_DFAUST.py` script. At the top of the script, change the path variables `SOURCE_DATA_PATH` and `SAVE_DATA_PATH`.  
`SOURCE_DATA_PATH` gives the path to your downloaded and unzipped source data contained in the folder "data" described above. 
`SAVE_DATA_PATH` specifies the location to save the preprocessed data; if it doesn't exist, it will be created along with two folders in it named ´train´ and ´test´. We use the DFaust train split for training and the MPI_Limits split for testing on out-of-distribution poses.

We use the same preprocessing strategy as [ArtEq](https://github.com/HavenFeng/ArtEq/tree/726287fcba0b8a1306b4370ec91661e236eb1909), and the preprocessing script has been created using functions from that repository.

#### Training
To train the rot. equivariant or standard model use the scripts `tasks/SemSeg/train_dfaust_rot.py` or `tasks/SemSeg/train_dfaust_standard.py`, respectively. An example is given here:
```bash
python train_dfaust_rot.py -conf_file confs/dfaust/dfaust_I_rot_pca_2F.yaml --gpu 0
```

```bash
python train_dfaust_standard.py --conf_file confs/dfaust/dfaust_I_standard.yaml --gpu 0
```
Trained models are saved in the ./logs folder; we use the model of the last epoch during testing to get the numbers reported in the paper.
#### Testing
For testing, you can use `tasks/SemSeg/test_dfaust_rot.py` and `tasks/SemSeg/test_dfaust_standard.py` and the `confs/dfaust/dfaust_test.yaml` config. You can specify the model(s) and number of frames to use for testing in those files (around line 200) by changing the lists
`model_paths =  ["path to your model dir/model_epoch_149.pth"]` and `n_frames_testing = [2]`. Then run e.g.:


```bash
python test_dfaust_rot.py --conf_file confs/dfaust/dfaust_test.yaml --gpu 0
```

```bash
python test_dfaust_standard.py --conf_file confs/dfaust/dfaust_test.yaml --gpu 0
```


### Scene understanding 
Train and test scripts, as well as configs, are saved under `tasks/SemSeg` and denoted with `scannet`.
#### Dataset
We use the [ScanNet](https://github.com/ScanNet/ScanNet) dataset segmenting into 20 classes. 
#### Training
Similar to the "human body parts segmentation" experiment, the relevant scripts are `tasks/SemSeg/train_scannet_rot.py` and `tasks/SemSeg/train_scannet_standard.py`. 
In `confs/scannet`, the collection of configs used is given. When using rotation, we fix the up vector and only sample one reference frame as described in the paper.

```bash
python train_scannet_rot.py -conf_file confs/scannet/dfaust_I_rot_pca_2F.yaml --gpu 0
```

```bash
python train_scannet_standard.py --conf_file confs/scannet/dfaust_I_standard.yaml --gpu 0
```
#### Testing
The following commands are examples of how to use the test scripts. Please make sure to use the right config with the right model. For example,  "I_SO2" means it expects a model that was trained without any rot augmentation and it will be tested with augmentation.

```bash
python test_scannet_rot.py --conf_file confs/scannet/scannet20_test_pca_I_SO2.yaml --saved_model path_to_your saved_rot_equ_model.pth --gpu 0 --save_output
```

```bash
python test_scannet_standard.py --conf_file confs/scannet/scannet20_test_standard_I_SO2.yaml --saved_model path_to_your_saved_standard_model.pth --gpu 0 --save_output
```

## 📝 TODOs
- [x] general code release
- [x] DFAUST train/test scripts and configs
- [x] ScanNet train/test scripts and configs
- [] Modelnet train/test scripts and configs

## 📚 BibTeX
If you find our code or paper useful, please cite:
```bibtex
@article{weijler2025roteq,
  title = {Efficient Continuous Group Convolutions for Local SE(3) Equivariance in 3D Point Clouds},
  author = {Weijler, L. and Hermosilla, P.},
  journal = {International Conference on 3D Vision (3DV)},
  year = {2025},
}
```

