"""Policy transforms for a Flexiv Rizon10 + Grav gripper driven in Cartesian EE-pose space.

The robot is trained from hand-held UMI gripper demonstrations, so there are no joints --
state and actions are both end-effector poses encoded with `openpi.shared.se3`:

    state[t]      absolute EE pose in the ROBOT BASE frame        (10,)
    actions[t+k]  EE pose relative to the pose at the CHUNK START (action_horizon, 10)

The dataset stores actions as *absolute* base-frame poses (the chunk start can be any frame,
so it cannot be baked in); `ChunkRelativePoseActions` converts them at load time and
`AbsolutePoseActions` undoes that at inference, so the policy server hands the robot absolute
base-frame poses it can servo to directly.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import se3


def make_rizon10_example() -> dict:
    """Creates a random input example for the Rizon10 policy."""
    rng = np.random.default_rng(0)
    pose = np.eye(4)
    pose[:3, 3] = rng.uniform(-0.5, 0.5, size=3)
    return {
        "observation/state": se3.mat_to_pose(pose, 0.0).astype(np.float32),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the tape",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Rizon10Inputs(transforms.DataTransformFn):
    """Converts Rizon10 observations into the format the model expects.

    Used for both training and inference.
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(data["observation/wrist_image"])

        # The UMI rig has a single wrist camera. It goes in the semantically correct slot
        # (`left_wrist_0_rgb`) and the other two slots are zero-padded and masked off.
        #
        # Caveat worth knowing if this underperforms: pi0.5 pretraining rarely masks
        # `base_0_rgb` off, so an all-zero masked base view is somewhat off-distribution.
        # The one-line fallback is to also put `wrist_image` in `base_0_rgb` with its mask
        # set to np.True_.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": np.zeros_like(wrist_image),
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                # We only mask padding images for pi0 models, not pi0-FAST.
                "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Rizon10Outputs(transforms.DataTransformFn):
    """Converts model outputs back to the Rizon10 action format. Inference only."""

    def __call__(self, data: dict) -> dict:
        # Strip the zero padding that took the actions up to the model action dimension.
        return {"actions": np.asarray(data["actions"][..., : se3.POSE_DIM])}


def _replace_pose(actions: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Writes `mats` back into the pose channels of `actions`, leaving the gripper alone."""
    out = np.array(actions, copy=True)
    out[..., : se3.GRIPPER_INDEX] = se3.mat_to_pose(mats, 0.0)[..., : se3.GRIPPER_INDEX]
    return out


@dataclasses.dataclass(frozen=True)
class ChunkRelativePoseActions(transforms.DataTransformFn):
    """Rewrites absolute EE-pose actions as poses relative to the chunk-start state.

    The SE(3) analogue of `transforms.DeltaActions`, which cannot be used here: it subtracts
    the state elementwise, which is meaningless for a 6-D rotation representation. The gripper
    channel stays absolute, matching the convention used everywhere else in openpi.
    """

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if "actions" not in data:
            return data
        ref = se3.pose_to_mat(np.asarray(data["state"])[..., : se3.POSE_DIM])
        abs_ = se3.pose_to_mat(np.asarray(data["actions"])[..., : se3.POSE_DIM])
        # `ref` gains a horizon axis so it broadcasts across the chunk.
        data["actions"] = _replace_pose(data["actions"], se3.relative(ref[..., None, :, :], abs_))
        return data


@dataclasses.dataclass(frozen=True)
class AbsolutePoseActions(transforms.DataTransformFn):
    """Composes chunk-relative EE-pose actions back onto the state. Inference only.

    The SE(3) analogue of `transforms.AbsoluteActions`. `Unnormalize` runs with strict=True
    over both "state" and "actions", so by the time this transform sees the data the state is
    the true unnormalized base-frame pose (zero-padded out to the model action dimension).
    """

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if "actions" not in data:
            return data
        ref = se3.pose_to_mat(np.asarray(data["state"])[..., : se3.POSE_DIM])
        rel = se3.pose_to_mat(np.asarray(data["actions"])[..., : se3.POSE_DIM])
        data["actions"] = _replace_pose(data["actions"], se3.compose(ref[..., None, :, :], rel))
        return data
