"""Convert UMI (hand-held gripper) MCAP recordings into a LeRobot dataset for pi0.5.

Each `.mcap` file is one episode holding three asynchronous streams:

    /camera/color/image   foxglove.CompressedImage   ~29 Hz   JPEG 640x480
    /optitrack/pose       foxglove.PoseInFrame      ~117 Hz   meters, quaternion XYZW
    /gripper_input        grumi.GripperInput         100 Hz   encoder_angle in DEGREES
                                                              (+ gamepad axes/buttons, dropped)

They get resampled onto a uniform grid with a zero-order hold (most recent sample), which is
what the policy actually sees online, and rewritten as:

    state[i]    absolute EE pose in the ROBOT BASE frame                       (10,)
    actions[i]  the same pose shifted forward by the measured actuation lag    (10,)

Actions are stored *absolute*; the chunk-start-relative conversion happens in the data loader
(`ChunkRelativePoseActions`), because the chunk start can be any frame.

Usage (from the repo root):

    uv pip install -r examples/umi_rizon10/requirements.in
    uv run examples/umi_rizon10/convert_mcap_to_lerobot.py \
        --data_dir /home/arthur/Downloads/umi_datasets/tape_pick_place_mcap \
        --task "pick up the tape and place it in the bin"

Add `--max_episodes 3` for a fast smoke run. The result is written under $HF_LEROBOT_HOME.
"""

import dataclasses
import logging
import pathlib
import shutil

import constants
import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro
import yaml

from openpi.shared import se3

logger = logging.getLogger(__name__)

NS_PER_S = 1_000_000_000


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Everything needed to map raw OptiTrack + encoder readings into robot-base EE poses."""

    t_ee_wrt_otbody: np.ndarray
    t_base_wrt_otworld: np.ndarray
    gripper_closed_angle_deg: float
    gripper_signed_range_deg: float
    latency_s: float

    @classmethod
    def load(cls, path: pathlib.Path) -> "Calibration":
        raw = yaml.safe_load(path.read_text())
        t_cam_wrt_otbody = np.asarray(raw["T_cam_wrt_otbody"], dtype=np.float64)
        t_cam_wrt_tcp = np.asarray(raw["T_cam_wrt_tcp"], dtype=np.float64)
        t_base_wrt_otworld = np.asarray(raw["T_base_wrt_otworld"], dtype=np.float64)

        if np.allclose(t_base_wrt_otworld, np.eye(4)):
            logger.warning(
                "T_base_wrt_otworld is the identity placeholder -- `state` will be in the raw "
                "OptiTrack world frame, NOT the robot base frame, and will not match what the "
                "robot reports at deploy time. Measure it and re-run this script. "
                "(Chunk-relative `actions` are unaffected.)"
            )
        if raw["gripper_signed_range_deg"] == 0:
            raise ValueError("gripper_signed_range_deg must be non-zero.")

        return cls(
            # The UMI rig is tracked as a rigid body; the camera is the bridge between the
            # tracked body and the Rizon10's TCP definition.
            t_ee_wrt_otbody=t_cam_wrt_otbody @ se3.invert(t_cam_wrt_tcp),
            t_base_wrt_otworld=t_base_wrt_otworld,
            gripper_closed_angle_deg=float(raw["gripper_closed_angle_deg"]),
            gripper_signed_range_deg=float(raw["gripper_signed_range_deg"]),
            latency_s=float(raw["latency_s"]),
        )

    def gripper_closedness(self, angle_deg: np.ndarray) -> np.ndarray:
        """Encoder degrees -> [0, 1] closedness (0 = open, 1 = closed), wrap-safe."""
        # Wrap into (-180, 180] relative to the closed angle first, so a reading that crossed
        # 360 degrees does not land at the opposite end of the range.
        delta = (np.asarray(angle_deg, dtype=np.float64) - self.gripper_closed_angle_deg + 180.0) % 360.0 - 180.0
        return np.clip(1.0 - delta / self.gripper_signed_range_deg, 0.0, 1.0)


def _read_episode(path: pathlib.Path) -> dict[str, tuple[np.ndarray, list]]:
    """Read one MCAP into `{topic: (log_times_ns, decoded_messages)}`, sorted by time."""
    buckets: dict[str, tuple[list[int], list]] = {
        constants.TOPIC_IMAGE: ([], []),
        constants.TOPIC_POSE: ([], []),
        constants.TOPIC_GRIPPER: ([], []),
    }
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, proto in reader.iter_decoded_messages(topics=list(buckets), log_time_order=True):
            times, msgs = buckets[channel.topic]
            times.append(message.log_time)
            # Only the payload we actually need is kept -- JPEG bytes stay compressed until
            # after resampling, so we decode ~1 frame in 1.5 instead of all of them.
            if channel.topic == constants.TOPIC_IMAGE:
                msgs.append(proto.data)
            elif channel.topic == constants.TOPIC_POSE:
                p, q = proto.pose.position, proto.pose.orientation
                msgs.append((p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            else:
                msgs.append(proto.encoder_angle[0])
    return {topic: (np.asarray(t, dtype=np.int64), m) for topic, (t, m) in buckets.items()}


def _resample_indices(times: dict[str, np.ndarray], fps: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a uniform grid and, per topic, the index of the most recent sample at each point.

    Zero-order hold rather than interpolation: at inference the policy is handed whatever the
    latest reading is, so training on held values matches deployment.
    """
    start = max(t[0] for t in times.values())
    end = min(t[-1] for t in times.values())
    step = NS_PER_S // fps
    grid = np.arange(start, end + 1, step, dtype=np.int64)
    # `side="right" - 1` is exactly "latest sample with timestamp <= grid point". The grid
    # starts at the newest first-timestamp, so the result is never negative.
    return grid, {topic: np.searchsorted(t, grid, side="right") - 1 for topic, t in times.items()}


def _decode_jpeg(payload: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode a JPEG frame.")
    return image_tools.resize_with_pad(bgr[..., ::-1], *constants.IMAGE_SIZE).astype(np.uint8)


def _build_episode(
    path: pathlib.Path, calib: Calibration, fps: int, shift: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list]:
    """Returns (state (N,10), actions (N,10), raw encoder degrees (N,), image indices, raw images)."""
    episode = _read_episode(path)
    times = {topic: t for topic, (t, _) in episode.items()}
    if any(len(t) < 2 for t in times.values()):
        raise ValueError(f"{path.name}: a topic has fewer than 2 messages.")

    _, indices = _resample_indices(times, fps)

    pose_rows = np.asarray([episode[constants.TOPIC_POSE][1][i] for i in indices[constants.TOPIC_POSE]])
    angles = np.asarray([episode[constants.TOPIC_GRIPPER][1][i] for i in indices[constants.TOPIC_GRIPPER]])

    # OptiTrack rigid body -> robot base.
    t_otbody = np.zeros((len(pose_rows), 4, 4))
    t_otbody[:, :3, :3] = se3.quat_xyzw_to_mat(pose_rows[:, 3:7])
    t_otbody[:, :3, 3] = pose_rows[:, 0:3]
    t_otbody[:, 3, 3] = 1.0
    t_ee_wrt_base = se3.invert(calib.t_base_wrt_otworld) @ t_otbody @ calib.t_ee_wrt_otbody

    state = se3.mat_to_pose(t_ee_wrt_base, calib.gripper_closedness(angles)).astype(np.float32)
    # The action for step i is the pose the gripper actually reached `shift` frames later --
    # that is what a command issued at step i produces once the robot's lag is accounted for.
    actions = state[np.minimum(np.arange(len(state)) + shift, len(state) - 1)]

    return state, actions, angles, list(indices[constants.TOPIC_IMAGE]), episode[constants.TOPIC_IMAGE][1]


def _report_gripper(angles_deg: np.ndarray, closedness: np.ndarray, calib: Calibration) -> None:
    """Print enough for a human to tell whether the gripper calibration is the right way round."""
    quantiles = [0, 5, 25, 50, 75, 95, 100]
    logger.info(
        "Raw encoder angle (deg), percentiles %s: %s",
        quantiles,
        np.round(np.percentile(angles_deg, quantiles), 2).tolist(),
    )
    logger.info(
        "Calibration says %.1f deg = closed, %.1f deg = open.",
        calib.gripper_closed_angle_deg,
        calib.gripper_closed_angle_deg + calib.gripper_signed_range_deg,
    )
    logger.info(
        "Resulting closedness in [0,1], percentiles %s: %s (mean %.3f)",
        quantiles,
        np.round(np.percentile(closedness, quantiles), 3).tolist(),
        closedness.mean(),
    )
    clipped = float(np.mean((closedness <= 0.0) | (closedness >= 1.0)))
    if clipped > 0.25:
        logger.warning(
            "%.0f%% of frames clip to 0 or 1 -- the encoder range likely extends past the "
            "calibrated [closed, open] interval. Widen gripper_signed_range_deg.",
            100 * clipped,
        )
    logger.warning(
        "Sanity-check the direction against the task: a pick-and-place should sit OPEN while "
        "approaching and CLOSED while transporting. If that is inverted, negate "
        "gripper_signed_range_deg and shift gripper_closed_angle_deg to the other end."
    )


def main(
    data_dir: str,
    task: str,
    *,
    repo_name: str = "umi/rizon10_tape_pick_place",
    calibration: str = str(pathlib.Path(__file__).parent / "calibration" / "rizon10_tape_pick_place.yaml"),
    fps: int = constants.FPS,
    latency_s: float | None = None,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    calib = Calibration.load(pathlib.Path(calibration))
    latency_s = calib.latency_s if latency_s is None else latency_s
    shift = round(latency_s * fps)
    logger.info("Latency %.3f s at %d Hz -> action shift of %d frame(s).", latency_s, fps, shift)
    if shift == 0:
        logger.warning(
            "Action shift rounded to 0 frames, so actions[0] will be identical to the state and "
            "the first relative action will be exactly identity. Raise --fps or --latency_s."
        )

    # Episode numbering is not contiguous (episode_192 is absent), so glob rather than count.
    paths = sorted(pathlib.Path(data_dir).glob("*.mcap"))
    if not paths:
        raise FileNotFoundError(f"No .mcap files under {data_dir}")
    if max_episodes is not None:
        paths = paths[:max_episodes]
    logger.info("Converting %d episode(s) from %s", len(paths), data_dir)

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=constants.ROBOT_TYPE,
        fps=fps,
        features=constants.features(),
        image_writer_threads=10,
        image_writer_processes=5,
    )

    all_angles: list[np.ndarray] = []
    all_closedness: list[np.ndarray] = []
    for path in tqdm.tqdm(paths, desc="episodes"):
        state, actions, angles, image_indices, raw_images = _build_episode(path, calib, fps, shift)
        all_angles.append(angles)
        all_closedness.append(state[:, se3.GRIPPER_INDEX])

        # Decode each distinct JPEG once -- at 20 Hz from a ~29 Hz camera the hold repeats frames.
        decoded: dict[int, np.ndarray] = {}
        for i, image_index in enumerate(image_indices):
            if image_index not in decoded:
                decoded[image_index] = _decode_jpeg(raw_images[image_index])
            dataset.add_frame(
                {
                    "wrist_image": decoded[image_index],
                    "state": state[i],
                    "actions": actions[i],
                    "task": task,
                }
            )
        dataset.save_episode()

    _report_gripper(np.concatenate(all_angles), np.concatenate(all_closedness), calib)

    if push_to_hub:
        dataset.push_to_hub(tags=["umi", "rizon10", "cartesian"], private=True, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)
