import random
import numpy as np
import torch
from typing import Optional, Union

# from pytorch3d import transforms
from einops import rearrange, repeat
import torch.nn.functional as F
from itertools import product

import torch_scatter

Device = Union[str, torch.device]


def all_index_combinations(n_A: int, n_B: int, device: Optional[Device] = None):
    # Generate all possible combinations of indices
    index_combinations = torch.tensor(
        list(product(range(n_A), range(n_B))), device=device
    )
    return index_combinations


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""

### Section Function copied from pytorch3D, due to installation issues on roberto
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device)[0]


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


### End Section

### Begin Section FAENet
"""
Functions from this sections are adapted
from https://faenet.readthedocs.io/en/latest/_modules/faenet/frame_averaging.html#frame_averaging_3D.

"""


def sample_global_reference_frames_pca(
    points: torch.Tensor,  # batchsize
    axis_fixed=False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:

    if axis_fixed is None or not axis_fixed:
        # calculate Cov matrix of each points neighborhood

        points_centered = points - points.mean(dim=1, keepdim=True)
        C = torch.einsum(
            "bij,bjk->bik", points_centered.transpose(1, 2), points_centered
        )

        eigenval, eigenvec = torch.linalg.eigh(C)

        # make all positive oriented
        eigenvec[torch.where(torch.linalg.det(eigenvec) < 0)] *= -1

        plus_minus_list = list(product([1, -1], repeat=3))
        # remove the ones where it would be -1 since we dont want reflection, should be 4 in total then
        plus_minus_list = [
            torch.tensor(x, device=device) for x in plus_minus_list if np.prod(x) == 1
        ]
        # random.shuffle(plus_minus_list)
        plus_minus_tensor = torch.stack(plus_minus_list, dim=0).unsqueeze(
            1
        )  # 4 x 1 x 3
        # repeat to have 4 frames for each point and multiply columns of all frames with either 1 or -1
        ref_frames = plus_minus_tensor * repeat(
            eigenvec.unsqueeze(1), "n k d1 d2 -> n (times k) d1 d2", times=4
        )
    else:

        raise NotImplementedError(
            "Sampling global ref frames with fixed axes is not implemented"
        )

    return rearrange(ref_frames, "n k d1 d2 -> n k (d1 d2)")  # n x 4 or 2 x 9


def sample_reference_frames_pca(
    points: torch.Tensor,  # n x 3
    p_neighborhood,
    axis_fixed=False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    mask = p_neighborhood.neighbors_[:, 1] < 0
    p_neighborhood.neighbors_[mask, 1] = p_neighborhood.neighbors_[
        mask, 0
    ]  # add self_loop if neighbors are missing

    neighborhood_matrices = rearrange(
        points[p_neighborhood.neighbors_[:, 1]], "(b n) d -> b n d", n=p_neighborhood.k_
    )

    if axis_fixed is None or not axis_fixed:
        # calculate Cov matrix of each points neighborhood

        neighborhood_matrices_centered = (
            neighborhood_matrices - neighborhood_matrices.mean(dim=1, keepdim=True)
        )
        C = torch.einsum(
            "bij,bjk->bik",
            neighborhood_matrices_centered.transpose(1, 2),
            neighborhood_matrices_centered,
        )

        # Eigendecomposition
        # returns eigenvec sorted in descending order (U), S eigenvals, V =U.t()
        # eigenval, eigenvec = torch.linalg.eigh(C) this is crap

        eigenval, eigenvec = torch.linalg.eigh(C)

        # make all positive oriented
        eigenvec[torch.where(torch.linalg.det(eigenvec) < 0)] *= -1

        plus_minus_list = list(product([1, -1], repeat=3))
        # remove the ones where it would be -1 since we dont want reflection, should be 4 in total then
        plus_minus_list = [
            torch.tensor(x, device=device) for x in plus_minus_list if np.prod(x) == 1
        ]
        # random.shuffle(plus_minus_list)
        plus_minus_tensor = torch.stack(plus_minus_list, dim=0).unsqueeze(
            1
        )  # 4 x 1 x 3
        # repeat to have 4 frames for each point and multiply columns of all frames with either 1 or -1
        ref_frames = plus_minus_tensor * repeat(
            eigenvec.unsqueeze(1), "n k d1 d2 -> n (times k) d1 d2", times=4
        )
    else:

        neighborhood_matrices[:, :, int(axis_fixed)] = 0
        neighborhood_matrices_centered = (
            neighborhood_matrices - neighborhood_matrices.mean(dim=1, keepdim=True)
        )
        C = torch.einsum(
            "bij,bjk->bik",
            neighborhood_matrices_centered.transpose(1, 2),
            neighborhood_matrices_centered,
        )
        # print(C)

        # Eigendecomposition
        # returns eigenvec sorted in descending order (U), S eigenvals, V =U.t()
        # eigenval, eigenvec = torch.linalg.eigh(C) this is crap

        eigenval, eigenvec = torch.linalg.eigh(C)
        # Sort, if necessary

        eigenvec = torch.flip(eigenvec, dims=[-1])
        eigenval = torch.flip(eigenval, dims=[-1])

        # make all positive oriented
        eigenvec[torch.where(torch.linalg.det(eigenvec) < 0)] *= -1

        plus_minus_list = [(1, 1, 1), (-1, -1, 1)]

        # remove the ones where it would be -1 since we dont want reflection, should be 4 in total then
        plus_minus_list = [
            torch.tensor(x, device=device) for x in plus_minus_list if np.prod(x) == 1
        ]
        # random.shuffle(plus_minus_list)
        plus_minus_tensor = torch.stack(plus_minus_list, dim=0).unsqueeze(
            1
        )  # 2 x 1 x 3
        # repeat to have 2 frames for each point and multiply columns of all frames with either 1 or -1
        ref_frames = plus_minus_tensor * repeat(
            eigenvec.unsqueeze(1), "n k d1 d2 -> n (times k) d1 d2", times=2
        )

        if axis_fixed == 0:
            ref_frames = ref_frames[:, :, :, [2, 0, 1]]

        elif axis_fixed == 1:
            ref_frames = ref_frames[:, :, :, [0, 2, 1]]
        EPS = 1e-06
        ref_frames = torch.where(torch.abs(ref_frames) < EPS, 0.0, ref_frames)

    return rearrange(ref_frames, "n k d1 d2 -> n k (d1 d2)")  # n x 4 or 2 x 9


### End Section


def random_rotate(p_hierarchy):
    rot_matrix = random_rotation(device=p_hierarchy.pcs_[0].pts_.device)
    for pc in p_hierarchy.pcs_:
        # points are row-wise:
        pc.pts_ = torch.matmul(pc.pts_, rot_matrix.transpose(1, 0))
        # frames are column-wise:
        transformed_frames = torch.einsum(
            "nm,ijml -> ijnl",
            rot_matrix,
            rearrange(pc.local_frames_, "n k (d1 d2) -> n k d1 d2", d1=3, d2=3),
        )
        pc.local_frames_ = rearrange(transformed_frames, "n l d1 d2 -> n l (d1 d2)")

    return p_hierarchy


def sample_reference_frames(
    n_origins: int,
    n_frames: int,
    axis_fixed: int = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """Sample random frames with uniform distribution.

    Args:
        n_origins (int): Number of origins (points).
        n_frames (int): Number of frames to be sampled per origin.

    Returns:
        torch.tensor: storing basis vectors of each sampled reference frame.
    """
    if axis_fixed is None or not axis_fixed:
        # use pytorch3d function ()
        rnd_rot_matrices = random_rotations(
            n_origins * n_frames, dtype=dtype, device=device
        )  # n_origins*n_frames x 3 x 3
        final_ref_frames = rearrange(
            rnd_rot_matrices, "(n k) d1 d2 -> n k (d1 d2)", n=n_origins, k=n_frames
        )  # n_origins x n_frames x 9
    else:
        random_angles = torch.rand(n_origins * n_frames, device=device) * 2 * np.pi
        zeros = torch.zeros_like(random_angles)
        ones = torch.ones_like(random_angles)
        # (self.max_angle_-self.min_angle_) + self.min_angle_
        # counter clockwise rotations for pre-multiplication of col vectors
        if axis_fixed == 0:
            rnd_rot_matrices = torch.stack(
                (
                    ones,
                    zeros,
                    zeros,
                    zeros,
                    torch.cos(random_angles),
                    -torch.sin(random_angles),
                    zeros,
                    torch.sin(random_angles),
                    torch.cos(random_angles),
                ),
                -1,
            )
        elif axis_fixed == 1:
            rnd_rot_matrices = torch.stack(
                (
                    torch.cos(random_angles),
                    zeros,
                    torch.sin(random_angles),
                    zeros,
                    ones,
                    zeros,
                    -torch.sin(random_angles),
                    zeros,
                    torch.cos(random_angles),
                ),
                -1,
            )
        elif axis_fixed == 2:
            rnd_rot_matrices = torch.stack(
                (
                    torch.cos(random_angles),
                    -torch.sin(random_angles),
                    zeros,
                    torch.sin(random_angles),
                    torch.cos(random_angles),
                    zeros,
                    zeros,
                    zeros,
                    ones,
                ),
                -1,
            )

        final_ref_frames = rearrange(
            rnd_rot_matrices, "(n k) d -> n k d", n=n_origins, k=n_frames
        )  # n_origins x n_frames x 9
    # column vectors of transformation matrices are new basis vectors of new reference frame
    return final_ref_frames


def get_single_relative_rot(
    frame_A: torch.Tensor, frame_B: torch.Tensor, return_representation: str = "matrix"
) -> torch.Tensor:
    """Get rotation matrix between frame A and B.

    Args:
        frames_A (torch.Tensor): shape 1 x 9,
                                 storing coord. axis of local reference frame.
        frames_B (torch.Tensor): shape 1 x 9,
                                storing coord. axis of local reference frame.

    Returns:
        torch.Tensor: relative rotation matrices between frame A and B

    """
    covered_representations = ["matrix", "6D", "quaternion"]
    if return_representation not in covered_representations:
        raise ValueError(
            f"Input to function parameter 'return_representation' = {return_representation}, \
                         not in covered representation types {covered_representations}"
        )

    frame_A_matrix = rearrange(frame_A, "(d1 d2) -> d1 d2", d1=3, d2=3)
    frame_B_matrix = rearrange(frame_B, "(d1 d2) -> d1 d2", d1=3, d2=3)

    # treat rot as basis change -> happens from B to A -> basis change matrix = A_inv * B
    frame_A_matrix_inverse = frame_A_matrix.transpose(
        1, 0
    )  # orthogonal matrix, inverse is transpose
    rel_rotmatrix = torch.matmul(frame_A_matrix_inverse, frame_B_matrix)
    if return_representation == "matrix":
        return rearrange(rel_rotmatrix, "d1 d2 -> (d1 d2)")  # 1 x 9
    if return_representation == "quaternion":
        return matrix_to_quaternion(rel_rotmatrix)
    if return_representation == "6D":
        return matrix_to_rotation_6d(rel_rotmatrix)


def get_relative_rot(
    frames_A: torch.Tensor,
    frames_B: torch.Tensor,
    return_representation: str = "matrix",
) -> torch.Tensor:
    """Get rotation matrix between frame A and B.

    Args:
        frames_A (torch.Tensor): shape n_origins x n_frames x 9,
                                 storing coord. axis of local reference frames.
        frames_B (torch.Tensor): shape n_origins x m_frames x 9,
                                storing coord. axis of local reference frames.

    Returns:
        torch.Tensor: relative rotation matrices between frames in A and B
                     -> in total n_frames_A* n_frames_B rotation matrices.

    """
    covered_representations = ["matrix", "6D", "quaternion"]
    if return_representation not in covered_representations:
        raise ValueError(
            f"Input to function parameter 'return_representation' = {return_representation}, \
                         not in covered representation types {covered_representations}"
        )

    frames_A_matrices = rearrange(frames_A, "n k (d1 d2) -> n k d1 d2", d1=3, d2=3)
    frames_B_matrices = rearrange(frames_B, "n k (d1 d2) -> n k d1 d2", d1=3, d2=3)

    # rearrange so we get all combinations
    n_frames_per_point_A = frames_A.shape[1]
    n_frames_per_point_B = frames_B.shape[1]

    frames_A_matrices = repeat(
        frames_A_matrices, "n k d1 d2 -> n (k times) d1 d2", times=n_frames_per_point_B
    )  # repeat each frame
    frames_B_matrices = repeat(
        frames_B_matrices, "n k d1 d2 -> n (times k) d1 d2", times=n_frames_per_point_A
    )  # repeat batch of frames

    # treat rot as basis change -> happens from B to A -> basis change matrix = A_inv * B
    frames_A_matrices_inverse = frames_A_matrices.transpose(
        2, 3
    )  # orthogonal matrix, inverse is transpose
    rel_rotmatrices = torch.matmul(frames_A_matrices_inverse, frames_B_matrices)
    if return_representation == "matrix":
        return rearrange(
            rel_rotmatrices, "n l d1 d2 -> n l (d1 d2)"
        )  # n_origins x (n_frames_A * n_frames_B) x 9
    if return_representation == "quaternion":
        return matrix_to_quaternion(rel_rotmatrices)
    if return_representation == "6D":
        return matrix_to_rotation_6d(rel_rotmatrices)
    if return_representation == "single_angle":  # only works for one axis fixed
        NotImplementedError("single angular value as rel rot not implemented yet")


def change_points_to_local_frame(
    points: torch.Tensor, origins: torch.Tensor, ref_frames: torch.Tensor
) -> torch.Tensor:
    """Get points in local reference frames.

    Args:
        points (torch.Tensor): coordinates of points in world coordinates.
        origins (torch.Tensor): coordinates of origins of local frames in world coordiantes.
        ref_frames (torch.Tensor): axes of local reference frames in world coordinates.

    Returns:
        torch.Tensor: coordinates of points in local reference frames.
    """

    # inverse translation
    points_translated = points - origins

    # inverse rotation
    frames_matrices = rearrange(ref_frames, "n k (d1 d2) -> n k d1 d2", d1=3, d2=3)
    frames_matrices_invers = frames_matrices.transpose(2, 3)

    points_translated = repeat(
        points_translated, "n d -> n k d", k=frames_matrices.size(1)
    )
    points_translated_cols = rearrange(points_translated, "n k (d x)-> n k d x", x=1)
    points_new_frame = torch.matmul(frames_matrices_invers, points_translated_cols)

    return rearrange(
        points_new_frame, "n k d x -> n k (d x)"
    )  # n_origins x n_frames x 9


def change_direction_to_local_frame(
    direction_vector: torch.Tensor, ref_frames: torch.Tensor
) -> torch.Tensor:
    """Get points in local reference frames.

    Args:
        points (torch.Tensor): coordinates of points in world coordinates.
        ref_frames (torch.Tensor): axes of local reference frames in world coordinates.

    Returns:
        torch.Tensor: coordinates of points in local reference frames.
    """
    # no inverse translation for chaning frames of direction vector

    # inverse rotation
    frames_matrices = rearrange(ref_frames, "n k (d1 d2) -> n k d1 d2", d1=3, d2=3)
    # frames_matrices_invers = frames_matrices.transpose(2,3)

    direction_vector = repeat(
        direction_vector, "n d -> n k d", k=frames_matrices.size(1)
    )
    # direction_vector_cols = rearrange(direction_vector, 'n k (d x)-> n k d x', x =1)
    # points_new_frame = torch.matmul(frames_matrices_invers, direction_vector_cols)
    points_new_frame = torch.matmul(
        direction_vector.unsqueeze(2), frames_matrices
    )  # nx kx 1 x 3 - rowwise points

    # return rearrange(points_new_frame,  'n k d x -> n k (d x)') # n_origins x n_frames x 3
    return points_new_frame.squeeze(2)
