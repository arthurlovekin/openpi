"""Run a trained pi0.5 policy on a Flexiv Rizon10 with a Grav gripper.

This is the runtime client, not the training loop -- it talks to a policy server started with
`scripts/serve_policy.py`. It runs in the ROBOT-SIDE environment (the one that has `flexivrdk`),
not the openpi training venv. See README.md for the environment split.

The policy server returns ABSOLUTE end-effector poses in the robot base frame -- the
chunk-relative -> absolute composition happens inside `AbsolutePoseActions` on the server -- so
all this script has to do is decode them and servo.

    python examples/umi_rizon10/main.py --robot_sn Rizon10-062394 \
        --gripper_name Grav --remote_host <policy-server-ip>
"""

import contextlib
import dataclasses
import logging
import signal
import time

import constants
import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

from openpi.shared import se3

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    # --- Hardware ---
    robot_sn: str = "Rizon10-062394"
    # Gripper name as configured in Flexiv Elements -> Settings -> Device.
    gripper_name: str = "Grav"
    # OpenCV device index for the wrist camera. See `WristCamera` -- swap this for the real
    # driver if the wrist camera is not a plain V4L2 device.
    camera_index: int = 0

    # --- Policy server ---
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000
    prompt: str = "pick up the tape and place it in the bin"

    # --- Rollout ---
    max_timesteps: int = 400
    # How many actions to execute from each predicted chunk before re-querying the server.
    # Half the action horizon, matching the DROID example's 8-of-16 ratio.
    open_loop_horizon: int = 8

    # --- Motion limits ---
    # NOTE: the human demonstrations reach ~0.6 m/s and ~70 deg/s at the 90th percentile
    # (max ~0.87 m/s, ~138 deg/s). The Flexiv NRT defaults are 0.5 m/s and 1.0 rad/s
    # (~57 deg/s), so the robot will lag behind fast chunks until these are raised. Start
    # conservative and increase deliberately.
    max_linear_vel: float = 0.25  # m/s
    max_angular_vel: float = 0.8  # rad/s
    # Safety guard: refuse to command a pose further than this from the current TCP. A wrong
    # T_base_wrt_otworld calibration shows up here as an immediate large jump.
    max_jump_m: float = 0.30

    # --- Gripper ---
    gripper_velocity: float = 0.1  # m/s
    gripper_force: float = 20.0  # N


class WristCamera:
    """Wrist camera capture.

    TODO(deploy): this assumes a plain V4L2 device. Replace with the actual wrist-camera
    driver -- it must be the SAME camera the `T_cam_wrt_tcp` extrinsics were measured for,
    otherwise the images will not match what the policy was trained on.
    """

    def __init__(self, index: int):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}.")

    def read_rgb(self) -> np.ndarray:
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError("Wrist camera read failed.")
        return np.ascontiguousarray(bgr[..., ::-1])

    def close(self) -> None:
        self._cap.release()


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Defer Ctrl+C until after the protected block, so it cannot kill a server call midway."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _connect(args: Args):
    import flexivrdk

    robot = flexivrdk.Robot(args.robot_sn)
    if robot.fault():
        logger.warning("Robot is in fault state, clearing.")
        robot.ClearFault()
    robot.Enable()
    while not robot.operational():
        time.sleep(1)

    gripper = flexivrdk.Gripper(robot)
    tool = flexivrdk.Tool(robot)
    gripper.Enable(args.gripper_name)
    # Switching the tool updates gravity compensation and, importantly, the TCP definition --
    # the frame every pose in the dataset is expressed in.
    tool.Switch(args.gripper_name)
    gripper.Init()
    while gripper.states().is_moving:
        time.sleep(0.5)

    mode = flexivrdk.Mode
    robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
    robot.ExecutePlan("PLAN-Home")
    while robot.busy():
        time.sleep(1)

    robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
    robot.ExecutePrimitive("ZeroFTSensor", {})
    while not robot.primitive_states()["terminated"]:
        time.sleep(1)

    robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
    robot.SetForceControlAxis([False] * 6)  # pure motion control
    return robot, gripper


def _observe(robot, gripper, camera: WristCamera, args: Args) -> tuple[dict, np.ndarray]:
    """Builds the policy request and returns it with the current TCP pose as a 4x4."""
    # Flexiv reports [x, y, z, qw, qx, qy, qz] in meters -- note WXYZ, whereas the OptiTrack
    # recordings the dataset came from were XYZW.
    tcp = se3.flexiv_to_mat(np.asarray(robot.states().tcp_pose))

    params = gripper.params()
    span = max(params.max_width - params.min_width, 1e-6)
    closedness = float(np.clip((params.max_width - gripper.states().width) / span, 0.0, 1.0))

    request = {
        "observation/wrist_image": image_tools.resize_with_pad(camera.read_rgb(), *constants.IMAGE_SIZE).astype(
            np.uint8
        ),
        "observation/state": se3.mat_to_pose(tcp, closedness).astype(np.float32),
        "prompt": args.prompt,
    }
    return request, tcp


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    robot, gripper = _connect(args)
    camera = WristCamera(args.camera_index)
    client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    params = gripper.params()

    chunk: np.ndarray | None = None
    executed = 0
    last_gripper_command: float | None = None
    period = 1.0 / constants.FPS

    try:
        for step in range(args.max_timesteps):
            loop_start = time.time()
            request, tcp = _observe(robot, gripper, camera, args)

            if chunk is None or executed >= args.open_loop_horizon:
                with prevent_keyboard_interrupt():
                    chunk = np.asarray(client.infer(request)["actions"])
                assert chunk.shape == (constants.ACTION_HORIZON, 10), chunk.shape
                executed = 0

            action = chunk[executed]
            executed += 1

            target = se3.pose_to_flexiv(action)
            jump = float(np.linalg.norm(target[:3] - tcp[:3, 3]))
            if jump > args.max_jump_m:
                raise RuntimeError(
                    f"Step {step}: commanded pose is {jump:.3f} m from the current TCP "
                    f"(limit {args.max_jump_m} m). Check T_base_wrt_otworld in the conversion "
                    f"calibration -- a wrong base frame looks exactly like this."
                )

            robot.SendCartesianMotionForce(
                target.tolist(), [0.0] * 6, [0.0] * 6, args.max_linear_vel, args.max_angular_vel
            )

            # Binarize the gripper, as the DROID example does, and only re-issue on a change --
            # Grav Move() calls block until the request is delivered.
            command = params.min_width if action[9] > 0.5 else params.max_width
            if command != last_gripper_command:
                gripper.Move(command, args.gripper_velocity, args.gripper_force)
                last_gripper_command = command

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            else:
                logger.warning("Step %d took %.1f ms, over the %.1f ms budget.", step, 1e3 * elapsed, 1e3 * period)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        robot.Stop()
        camera.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
