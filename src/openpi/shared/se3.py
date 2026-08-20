"""SE(3) helpers for Cartesian end-effector pose action spaces.

Poses are encoded as a 10-D vector::

    [x, y, z, r11, r21, r31, r12, r22, r32, gripper]
     └─ 3 ─┘  └─ first two COLUMNS of R ─┘  └──1──┘

The rotation uses the continuous 6-D representation of Zhou et al. (2019) -- the first
two columns of the rotation matrix, decoded with Gram-Schmidt. It has no wrapping
discontinuity, which is why it is preferred over Euler angles or rotation vectors for
regression targets.

The gripper channel is a normalized closedness in ``[0, 1]`` (``0`` = open, ``1`` = closed),
matching the openpi convention.

All functions accept arbitrary leading batch dimensions.
"""

import numpy as np

# Length of the encoded pose vector, and the index of the gripper channel within it.
POSE_DIM = 10
GRIPPER_INDEX = 9

# Names of the 10 channels, used for the LeRobot feature schema.
POSE_NAMES = ("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32", "gripper")


def _normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12, None)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Decode the 6-D rotation representation into a proper rotation matrix.

    Args:
        rot6d: `(..., 6)` array holding the first two columns of R, concatenated.

    Returns:
        `(..., 3, 3)` orthonormal, right-handed rotation matrices.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    c1, c2 = rot6d[..., 0:3], rot6d[..., 3:6]
    b1 = _normalize(c1)
    # Gram-Schmidt: remove the b1 component from c2 before normalizing.
    b2 = _normalize(c2 - np.sum(b1 * c2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Encode a rotation matrix as its first two columns."""
    mat = np.asarray(mat, dtype=np.float64)
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Decode a `(..., 10)` pose vector into `(..., 4, 4)` homogeneous transforms.

    The gripper channel is ignored.
    """
    pose = np.asarray(pose, dtype=np.float64)
    mat = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.float64)
    mat[..., :3, :3] = rot6d_to_mat(pose[..., 3:9])
    mat[..., :3, 3] = pose[..., 0:3]
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose(mat: np.ndarray, gripper: np.ndarray | float) -> np.ndarray:
    """Encode `(..., 4, 4)` transforms plus a gripper channel into `(..., 10)` poses."""
    mat = np.asarray(mat, dtype=np.float64)
    gripper = np.broadcast_to(np.asarray(gripper, dtype=np.float64), mat.shape[:-2])
    return np.concatenate([mat[..., :3, 3], mat_to_rot6d(mat[..., :3, :3]), gripper[..., None]], axis=-1)


def invert(mat: np.ndarray) -> np.ndarray:
    """Invert `(..., 4, 4)` rigid transforms in closed form."""
    mat = np.asarray(mat, dtype=np.float64)
    rot_t = np.swapaxes(mat[..., :3, :3], -1, -2)
    out = np.zeros_like(mat)
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", rot_t, mat[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def relative(ref: np.ndarray, abs_: np.ndarray) -> np.ndarray:
    """Express `abs_` in the frame of `ref`: ``inv(ref) @ abs_``."""
    return np.einsum("...ij,...jk->...ik", invert(ref), np.asarray(abs_, dtype=np.float64))


def compose(ref: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Inverse of `relative`: ``ref @ rel``."""
    return np.einsum("...ij,...jk->...ik", np.asarray(ref, dtype=np.float64), np.asarray(rel, dtype=np.float64))


def _quat_to_mat(w, x, y, z) -> np.ndarray:
    norm = np.clip(np.sqrt(w * w + x * x + y * y + z * z), 1e-12, None)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def quat_xyzw_to_mat(quat: np.ndarray) -> np.ndarray:
    """`(..., 4)` quaternion in **XYZW** order (OptiTrack / Foxglove) -> `(..., 3, 3)`."""
    quat = np.asarray(quat, dtype=np.float64)
    return _quat_to_mat(quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2])


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    """`(..., 4)` quaternion in **WXYZ** order (Flexiv RDK) -> `(..., 3, 3)`."""
    quat = np.asarray(quat, dtype=np.float64)
    return _quat_to_mat(quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3])


def slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, weight: np.ndarray | float) -> np.ndarray:
    """Shortest-arc spherical linear interpolation between **XYZW** quaternions.

    Args:
        q0: `(..., 4)` quaternions reached at ``weight == 0``.
        q1: `(..., 4)` quaternions reached at ``weight == 1``.
        weight: broadcastable to the leading `(...)` shape. Values outside `[0, 1]` extrapolate
            along the same geodesic.

    Returns:
        `(..., 4)` unit quaternions, XYZW.

    `q` and `-q` are the same rotation, so `q1` is negated wherever the dot product is negative;
    without that the interpolation would take the long way round the sphere. Unlike a
    component-wise lerp this has constant angular velocity, which is what makes it the right
    thing for resampling a pose stream in time.
    """
    q0 = _normalize(np.asarray(q0, dtype=np.float64))
    q1 = _normalize(np.asarray(q1, dtype=np.float64))
    weight = np.asarray(weight, dtype=np.float64)[..., None]

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)

    theta = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))
    sin_theta = np.sin(theta)
    # Both branches are evaluated unconditionally, so the near-parallel one must stay finite even
    # where its result is thrown away.
    slerped = (np.sin((1.0 - weight) * theta) * q0 + np.sin(weight * theta) * q1) / np.clip(sin_theta, 1e-8, None)
    # Below ~1e-8 rad the chord and the arc agree to well past float64 precision, and the
    # shortest-arc flip above guarantees the sum is nowhere near zero.
    nlerped = _normalize(q0 + weight * (q1 - q0))
    return np.where(sin_theta < 1e-8, nlerped, slerped)


def mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """`(..., 3, 3)` or `(..., 4, 4)` rotation -> `(..., 4)` quaternion in **WXYZ** order.

    Uses the standard four-branch formulation, selecting the branch with the largest
    denominator so the result stays well conditioned near 180 degree rotations. The sign is
    canonicalized to ``w >= 0``.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape[-1] == 4:
        mat = mat[..., :3, :3]
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

    def _branch(sq, w, x, y, z):
        # `sq` is 4*q_i^2 for the branch's pivot component; clamp so unselected branches
        # (where it may be negative) produce finite garbage rather than NaN.
        s = 2.0 * np.sqrt(np.clip(sq, 1e-12, None))
        return np.stack([w(s), x(s), y(s), z(s)], axis=-1)

    cands = np.stack(
        [
            _branch(  # trace > 0
                1.0 + m00 + m11 + m22,
                lambda s: 0.25 * s,
                lambda s: (m21 - m12) / s,
                lambda s: (m02 - m20) / s,
                lambda s: (m10 - m01) / s,
            ),
            _branch(  # m00 is the largest diagonal entry
                1.0 + m00 - m11 - m22,
                lambda s: (m21 - m12) / s,
                lambda s: 0.25 * s,
                lambda s: (m01 + m10) / s,
                lambda s: (m02 + m20) / s,
            ),
            _branch(  # m11 largest
                1.0 + m11 - m00 - m22,
                lambda s: (m02 - m20) / s,
                lambda s: (m01 + m10) / s,
                lambda s: 0.25 * s,
                lambda s: (m12 + m21) / s,
            ),
            _branch(  # m22 largest
                1.0 + m22 - m00 - m11,
                lambda s: (m10 - m01) / s,
                lambda s: (m02 + m20) / s,
                lambda s: (m12 + m21) / s,
                lambda s: 0.25 * s,
            ),
        ],
        axis=-2,
    )

    trace = m00 + m11 + m22
    diag_argmax = np.argmax(np.stack([m00, m11, m22], axis=-1), axis=-1)
    index = np.where(trace > 0.0, 0, 1 + diag_argmax)
    quat = np.take_along_axis(cands, index[..., None, None].repeat(4, axis=-1), axis=-2)[..., 0, :]
    quat = _normalize(quat)
    # Canonicalize the sign so that q and -q (the same rotation) compare equal.
    return np.where(quat[..., :1] < 0.0, -quat, quat)


def pose_to_flexiv(pose: np.ndarray) -> np.ndarray:
    """`(..., 10)` pose -> Flexiv RDK `(..., 7)` pose `[x, y, z, qw, qx, qy, qz]` in meters."""
    mat = pose_to_mat(pose)
    return np.concatenate([mat[..., :3, 3], mat_to_quat_wxyz(mat)], axis=-1)


def flexiv_to_mat(pose7: np.ndarray) -> np.ndarray:
    """Flexiv RDK `(..., 7)` pose `[x, y, z, qw, qx, qy, qz]` -> `(..., 4, 4)`."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    mat = np.zeros((*pose7.shape[:-1], 4, 4), dtype=np.float64)
    mat[..., :3, :3] = quat_wxyz_to_mat(pose7[..., 3:7])
    mat[..., :3, 3] = pose7[..., 0:3]
    mat[..., 3, 3] = 1.0
    return mat
