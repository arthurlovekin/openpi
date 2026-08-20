import numpy as np

from openpi.shared import se3


def _random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniformly random rotation matrices via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.standard_normal((n, 3, 3)))
    # Fix the sign ambiguity of QR, then ensure a right-handed frame.
    q = q * np.sign(np.diagonal(r, axis1=-2, axis2=-1))[:, None, :]
    q[np.linalg.det(q) < 0, :, 0] *= -1
    return q


def _random_transforms(rng: np.random.Generator, n: int) -> np.ndarray:
    mat = np.zeros((n, 4, 4))
    mat[:, :3, :3] = _random_rotations(rng, n)
    mat[:, :3, 3] = rng.uniform(-1.0, 1.0, size=(n, 3))
    mat[:, 3, 3] = 1.0
    return mat


def test_pose_roundtrip():
    rng = np.random.default_rng(0)
    mat = _random_transforms(rng, 64)
    gripper = rng.uniform(0.0, 1.0, size=64)

    pose = se3.mat_to_pose(mat, gripper)
    assert pose.shape == (64, se3.POSE_DIM)
    np.testing.assert_allclose(pose[:, se3.GRIPPER_INDEX], gripper)
    np.testing.assert_allclose(se3.pose_to_mat(pose), mat, atol=1e-12)


def test_gram_schmidt_orthonormalizes_noisy_input():
    rng = np.random.default_rng(1)
    mat = _random_transforms(rng, 64)
    pose = se3.mat_to_pose(mat, 0.0)
    pose[:, 3:9] += rng.normal(scale=0.05, size=(64, 6))

    rot = se3.rot6d_to_mat(pose[:, 3:9])
    np.testing.assert_allclose(np.einsum("nij,nik->njk", rot, rot), np.broadcast_to(np.eye(3), (64, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot), np.ones(64), atol=1e-12)


def test_invert_and_relative_compose_roundtrip():
    rng = np.random.default_rng(2)
    ref = _random_transforms(rng, 32)
    abs_ = _random_transforms(rng, 32)

    np.testing.assert_allclose(se3.compose(ref, se3.invert(ref)), np.broadcast_to(np.eye(4), (32, 4, 4)), atol=1e-12)
    np.testing.assert_allclose(se3.compose(ref, se3.relative(ref, abs_)), abs_, atol=1e-12)


def test_relative_is_invariant_to_a_fixed_left_transform():
    """The property the whole chunk-relative action space rests on."""
    rng = np.random.default_rng(3)
    ref, abs_ = _random_transforms(rng, 16), _random_transforms(rng, 16)
    fixed = _random_transforms(rng, 1)[0]

    np.testing.assert_allclose(se3.relative(fixed @ ref, fixed @ abs_), se3.relative(ref, abs_), atol=1e-12)


def test_quaternion_orders_agree():
    rng = np.random.default_rng(4)
    quat_wxyz = se3.mat_to_quat_wxyz(_random_rotations(rng, 128))
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=-1)

    np.testing.assert_allclose(se3.quat_wxyz_to_mat(quat_wxyz), se3.quat_xyzw_to_mat(quat_xyzw), atol=1e-12)


def test_mat_to_quat_roundtrip_covers_all_branches():
    rng = np.random.default_rng(5)
    # Include exact 180 degree rotations about each axis -- trace = -1 there, which is
    # precisely where the trace branch degenerates and one of the diagonal branches must win.
    half_turns = np.stack([np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])])
    rot = np.concatenate([_random_rotations(rng, 512), np.eye(3)[None], half_turns])

    quat = se3.mat_to_quat_wxyz(rot)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-9)
    np.testing.assert_allclose(se3.quat_wxyz_to_mat(quat), rot, atol=1e-9)


def _quat_xyzw(rot: np.ndarray) -> np.ndarray:
    quat = se3.mat_to_quat_wxyz(rot)
    return np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)


def _geodesic_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle of the rotation taking `a` to `b`, for `(..., 3, 3)` rotation matrices."""
    trace = np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def test_slerp_hits_the_endpoints_at_constant_angular_velocity():
    rng = np.random.default_rng(7)
    r0, r1 = _random_rotations(rng, 64), _random_rotations(rng, 64)
    q0, q1 = _quat_xyzw(r0), _quat_xyzw(r1)

    # Compare rotations rather than quaternions: q and -q are the same rotation.
    np.testing.assert_allclose(se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(q0, q1, 0.0)), r0, atol=1e-9)
    np.testing.assert_allclose(se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(q0, q1, 1.0)), r1, atol=1e-9)

    # Constant angular velocity is the whole point: the midpoint is equidistant from both ends,
    # and each half is exactly half the total. A component-wise lerp would fail this.
    mid = se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(q0, q1, 0.5))
    total = _geodesic_angle(r0, r1)
    np.testing.assert_allclose(_geodesic_angle(r0, mid), total / 2, atol=1e-9)
    np.testing.assert_allclose(_geodesic_angle(mid, r1), total / 2, atol=1e-9)

    quarter = se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(q0, q1, 0.25))
    np.testing.assert_allclose(_geodesic_angle(r0, quarter), total / 4, atol=1e-9)


def test_slerp_takes_the_shortest_arc():
    """Negating an input is a no-op, because `q` and `-q` are the same rotation."""
    rng = np.random.default_rng(8)
    q0, q1 = _quat_xyzw(_random_rotations(rng, 64)), _quat_xyzw(_random_rotations(rng, 64))
    w = rng.uniform(0.0, 1.0, size=64)

    np.testing.assert_allclose(
        se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(q0, -q1, w)),
        se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(q0, q1, w)),
        atol=1e-9,
    )
    # The interpolated rotation never travels further than the direct geodesic.
    mid = se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(q0, q1, w))
    total = _geodesic_angle(se3.quat_xyzw_to_mat(q0), se3.quat_xyzw_to_mat(q1))
    assert np.all(_geodesic_angle(se3.quat_xyzw_to_mat(q0), mid) <= total + 1e-9)
    assert np.all(total <= np.pi + 1e-9)


def test_slerp_is_bi_invariant():
    """Licenses interpolating a pose stream in its raw tracker frame and transforming after."""
    rng = np.random.default_rng(9)
    r0, r1 = _random_rotations(rng, 32), _random_rotations(rng, 32)
    left, right = _random_rotations(rng, 1)[0], _random_rotations(rng, 1)[0]
    w = rng.uniform(0.0, 1.0, size=32)

    direct = left @ se3.quat_xyzw_to_mat(se3.slerp_quat_xyzw(_quat_xyzw(r0), _quat_xyzw(r1), w)) @ right
    transformed = se3.quat_xyzw_to_mat(
        se3.slerp_quat_xyzw(_quat_xyzw(left @ r0 @ right), _quat_xyzw(left @ r1 @ right), w)
    )
    np.testing.assert_allclose(direct, transformed, atol=1e-9)


def test_slerp_handles_near_identical_and_unbatched_inputs():
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    out = se3.slerp_quat_xyzw(quat, quat, 0.5)
    assert out.shape == (4,)
    np.testing.assert_allclose(out, quat, atol=1e-12)

    # The degenerate branch: closer together than any real 117 Hz tracker sample pair.
    nudged = se3.slerp_quat_xyzw(quat, quat + np.array([1e-12, 0.0, 0.0, 0.0]), 0.5)
    assert np.all(np.isfinite(nudged))
    np.testing.assert_allclose(np.linalg.norm(nudged), 1.0, atol=1e-12)


def test_flexiv_roundtrip():
    rng = np.random.default_rng(6)
    mat = _random_transforms(rng, 64)
    pose = se3.mat_to_pose(mat, 0.5)

    pose7 = se3.pose_to_flexiv(pose)
    assert pose7.shape == (64, 7)
    np.testing.assert_allclose(pose7[:, :3], mat[:, :3, 3], atol=1e-12)
    np.testing.assert_allclose(se3.flexiv_to_mat(pose7), mat, atol=1e-9)


def test_unbatched_inputs():
    mat = np.eye(4)
    pose = se3.mat_to_pose(mat, 1.0)
    assert pose.shape == (se3.POSE_DIM,)
    np.testing.assert_allclose(se3.pose_to_mat(pose), mat, atol=1e-12)
    np.testing.assert_allclose(se3.mat_to_quat_wxyz(mat), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
