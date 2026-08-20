"""Shared constants for the UMI -> Flexiv Rizon10 pipeline.

Imported by both `convert_mcap_to_lerobot.py` (which runs in the openpi venv) and
`main.py` (which runs in the robot-side env that has `flexivrdk`).
"""

from openpi.shared import se3

# --- Dataset -----------------------------------------------------------------------------

# Resampling rate of the LeRobot dataset. The MCAP camera stream runs at ~29 Hz, the gripper
# encoder at 100 Hz and OptiTrack at ~117 Hz; 20 Hz is a safe common grid and is comfortably
# inside the 1-100 Hz range the Flexiv Python RDK accepts for NRT Cartesian commands.
FPS = 20

# Must match `Pi0Config.action_horizon` in the `pi05_umi_rizon10` TrainConfig.
# 16 frames at 20 Hz = 0.80 s.
ACTION_HORIZON = 16

# Measured command->motion lag of the Rizon10. Actions are the state shifted forward by this
# much, so the pose recorded at t + LATENCY_S supervises the command issued at t.
LATENCY_S = 0.110

# pi0.5 resizes every image to 224x224 with `resize_with_pad`. We store the dataset already
# resized so that the training pixels are bit-identical to what `main.py` sends at deploy time.
IMAGE_SIZE = (224, 224)

# --- MCAP topics -------------------------------------------------------------------------

TOPIC_IMAGE = "/camera/color/image"  # foxglove.CompressedImage, JPEG 640x480
TOPIC_POSE = "/optitrack/pose"  # foxglove.PoseInFrame, meters, quaternion XYZW
TOPIC_GRIPPER = "/gripper_input"  # grumi.GripperInput, encoder_angle in DEGREES

# --- LeRobot feature schema ----------------------------------------------------------------

ROBOT_TYPE = "rizon10"


def features() -> dict:
    pose_names = list(se3.POSE_NAMES)
    return {
        "wrist_image": {
            "dtype": "image",
            "shape": (*IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (se3.POSE_DIM,), "names": pose_names},
        "actions": {"dtype": "float32", "shape": (se3.POSE_DIM,), "names": pose_names},
    }
