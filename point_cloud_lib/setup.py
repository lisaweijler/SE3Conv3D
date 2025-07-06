import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"

# Force use of PyTorch's internal CUDA toolkit
os.environ["CUDA_HOME"] = CUDA_HOME or ""
os.environ["CUDACXX"] = os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")

setup(
    name="point_cloud_lib",
    ext_modules=[
        CUDAExtension(
            name="point_cloud_lib_ops",
            sources=[
                "custom_ops/feature_aggregation/feat_basis_proj.cu",
                "custom_ops/feature_aggregation/feat_basis_proj_grads.cu",
                "custom_ops/ball_query/ball_query.cu",
                "custom_ops/ball_query/compute_keys.cu",
                "custom_ops/ball_query/build_grid_ds.cu",
                "custom_ops/ball_query/count_neighbors.cu",
                "custom_ops/ball_query/store_neighbors.cu",
                "custom_ops/ball_query/find_ranges_grid_ds.cu",
                "custom_ops/knn_query/knn_query.cu",
                "custom_ops/ops_list.cpp",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=[
        "point_cloud_lib.custom_ops",
        "point_cloud_lib",
        "point_cloud_lib.pc",
        "point_cloud_lib.augment",
        "point_cloud_lib.layers",
        "point_cloud_lib.metrics",
        "point_cloud_lib.data_sets",
        "point_cloud_lib.data_sets.loaders",
        "point_cloud_lib.utils",
    ],
    package_data={"point_cloud_lib_ops": ["*.so"]},
)
