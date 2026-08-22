# UMI → Flexiv Rizon10 (Cartesian end-effector pose)

Fine-tune π0.5 on hand-held [UMI](https://umi-gripper.github.io/) gripper demonstrations and
deploy on a Flexiv Rizon10 with a Grav gripper, operating in **Cartesian EE-pose space**
rather than joint space.

## What makes this different from the other examples


|              | libero / DROID                | here                                               |
| ------------ | ----------------------------- | -------------------------------------------------- |
| action space | joint velocity / OSC deltas   | **SE(3) end-effector pose**                        |
| state        | joint or EE pose, absolute    | EE pose, absolute, **robot base frame**            |
| actions      | absolute or elementwise delta | pose **relative to the chunk-start state**         |
| rotation     | axis-angle                    | first two **columns** of the rotation matrix (6-D) |


Both state and actions are 10-D (`openpi.shared.se3`):

```
[ x, y, z, r11, r21, r31, r12, r22, r32, gripper ]
   └ 3 ┘  └─ first two COLUMNS of R ──┘  └1┘
```

The 6-D rotation representation has no wrapping discontinuity, and Gram–Schmidt decoding
absorbs small regression errors back onto SO(3). The gripper channel is a normalized
closedness in `[0, 1]` (`0` = open, `1` = closed), the openpi convention.

### State vs. action, and why they are in different frames

A UMI recording has no notion of "state" versus "action" — only the tracked gripper pose. However, on the real robot the state of the robot will always lag behind the action command by some latency, since the signal takes time to get from the inference server to the robot machine, and the robot takes time to actually achieve the commanded pose. Mathematically, `s(t)=a(t-dt)` (or equivalently `a(t) = s(t+dt)`). We synthesize the split:

- **state** = the EE pose at time `t`, **absolute in the robot base frame**. This is "where am
I in the workspace".
- **actions** = the pose at `t + latency`, expressed **relative to the pose at the chunk
start**. This is "how do I move from here". The pose the gripper actually reached is the right
supervision target for a command issued at `t`.

`latency` = `robot_motion_lag_s` + `downlink_lag_s` from the calibration file — the Rizon10's own
command→motion lag (110 ms) plus the measured inference-server→robot downlink (68 ms), 178 ms in
total. It is **not** a whole number of 20 Hz frames (178 ms is 3.56 of them), so `a(t) = s(t+dt)`
is evaluated by interpolating the *raw* streams — OptiTrack at ~117 Hz, the encoder at 100 Hz —
at exactly `t + dt`: lerp position, SLERP rotation, lerp gripper closedness. Rounding the
lookahead to 4 frames instead would label every action for a 200 ms lag, worth a few mm of
position bias at the median and ~15 mm at the extremes, and would make `latency` a step function
that ignores any measurement change under 25 ms. The cost is that the last `latency` of each
episode is dropped, since its lookahead falls past the end of the recording; nothing real is
lost, because an event at `T` still supervises the frame at `T - latency`.

The relative step cannot be baked into the dataset, because the chunk start can be any frame.
The dataset stores absolute actions and `ChunkRelativePoseActions` converts them in the data
loader; `AbsolutePoseActions` undoes it at inference, so **the policy server returns absolute
base-frame poses** that `main.py` can servo to directly.

A useful consequence: chunk-relative actions are *invariant* to the robot-base ↔ OptiTrack
calibration (`inv(T_s) @ T_a` cancels any fixed left-multiplication). Only `state` depends on
it, so re-converting after measuring that transform changes nothing else.

> Note on `openpi.transforms.DeltaActions` / `AbsoluteActions`: those subtract the state
> elementwise, which is meaningless for a 6-D rotation representation. That is why this example
> ships SE(3) counterparts instead of reusing them.



## Environments

Two different environments are involved.

**Conversion + training** — the openpi venv, plus the MCAP reader:

```bash
uv pip install -r examples/umi_rizon10/requirements.in
```

**Robot runtime** (`main.py`) — the environment that has `flexivrdk` (on this machine, the
`flexiv_zed` conda env). It needs `openpi-client`, `numpy`, `opencv-python`, `tyro`, and
`openpi.shared.se3`. The `openpi` and `openpi.shared` packages have empty `__init__.py` files
and `se3.py` is pure numpy, so putting the source tree on `PYTHONPATH` is enough — no need to
install openpi's JAX stack on the robot machine:

```bash
conda activate flexiv_zed
pip install openpi-client tyro
PYTHONPATH=/path/to/openpi/src python examples/umi_rizon10/main.py --robot_sn Rizon10-062394
```



## 1. Calibration

Edit `calibration/rizon10_tape_pick_place.yaml`. Two entries are marked `MEASURE ME`:

- `T_base_wrt_otworld` — robot base pose in the OptiTrack world frame. Ships as identity;
the converter warns loudly. Until it is measured, `state` is in the raw OptiTrack world frame
(positions around `z ≈ 4.6 m`) and will **not** match what the robot reports at deploy time.
- `gripper_closed_angle_deg` **/** `gripper_signed_range_deg` — the encoder calibration, in
degrees. The shipped values (36° closed, 90° open) are *inferred from the data*, not
measured: the encoder piles up hard at 35.97–36.3° (a mechanical stop), clusters at 48–51°
(jaws held apart by the tape roll), and reaches 83–90.6°. Verify against the hardware; if it
is backwards, negate `gripper_signed_range_deg` and move `gripper_closed_angle_deg` to the
other end. The converter prints both distributions on every run.

`T_cam_wrt_otbody` and `T_cam_wrt_tcp` are already filled in from the earlier calibration run.

## 2. Convert MCAP → LeRobot

LeRobot identifies datasets by `repo_id` and looks them
up under `$HF_LEROBOT_HOME/<repo_id>`. If that folder is missing, the loader hits the
Hugging Face Hub and fails with a 401 for a local-only dataset. **Set this to the same
value for convert, norm stats, and train** (put it in your shell rc or tmux session):

```bash
export HF_LEROBOT_HOME=/mnt/data3/umi_datasets/lerobot_pi
```

```bash
uv run examples/umi_rizon10/convert_mcap_to_lerobot.py \
    --data_dir /mnt/data3/umi_datasets/tape_pick_place_mcap \
    --task "pick up the tape and place it in the bin"
```

Add `--max_episodes 3` for a fast smoke run. The dataset lands at
`$HF_LEROBOT_HOME/umi/rizon10_tape_pick_place` (default home is
`~/.cache/huggingface/lerobot` if you skip the export).

What it does:

- Reads `/camera/color/image` (JPEG 640×480, ~29 Hz), `/optitrack/pose` (m, quaternion
**XYZW**, ~117 Hz) and `/gripper_input` (`encoder_angle` in **degrees**, 100 Hz). Gamepad
`axes`/`buttons` are dropped.
- Resamples `state` onto a **20 Hz** grid with a zero-order hold (most recent sample) — held
values are what the policy sees online. `actions` are the deliberate exception: they are a label
with no online counterpart, so they are interpolated at the exact lookahead (above) instead of
being snapped to the grid. 20 Hz is also comfortably inside the 1–100 Hz range the Flexiv Python
RDK accepts.
- Maps poses into the robot base frame:
`T_ee_wrt_base = inv(T_base_wrt_otworld) @ T_otbody_wrt_otworld @ T_cam_wrt_otbody @ inv(T_cam_wrt_tcp)`.
- Writes `wrist_image` **already resized to 224×224 with** `resize_with_pad`, so training
pixels are bit-identical to what `main.py` sends.
- Writes `task` per frame, so `prompt_from_task=True` works.

Episode files are globbed rather than counted — the numbering is not contiguous
(`episode_192` is absent from this corpus).

## 3. Norm stats

```bash
export HF_LEROBOT_HOME=/mnt/data3/umi_datasets/lerobot_pi   # same as convert
uv run scripts/compute_norm_stats.py --config-name pi05_umi_rizon10
```

Fresh stats only — the pretrained `ur5e` / DROID assets describe **joint** spaces and are
meaningless here. This runs the real repack → `Rizon10Inputs` → `ChunkRelativePoseActions`
chain, so it doubles as an end-to-end check of the frame math. Inspect the result:

- `actions` rotation channels should have **mean** `r11 ≈ r22 ≈ 1` and off-diagonals **≈ 0** —
chunk-relative rotations centred on identity.
- `actions` translation means should be **≈ 0**.
- `state` translations should span the physical workspace.

Anything else means the frame chain is wrong.

## 4. Train

```bash
export HF_LEROBOT_HOME=/mnt/data3/umi_datasets/lerobot_pi   # same as convert
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_umi_rizon10 \
    --exp-name=my_experiment --overwrite
```

`pi05_umi_rizon10` uses `action_horizon=16` (0.80 s at 20 Hz) and `batch_size=32`. The dataset
is small — 195 episodes, ~23 min, ~27k frames — so it deliberately departs from `pi05_libero`'s
`batch_size=256` and 10k-step warmup, which would leave the LR still ramping for a third of a
30k-step run.

## 5. Serve and run

Three terminals: the GPU box runs the policy server, the robot laptop forwards a local
port to that server over SSH, then the robot client talks to `localhost`.

The server binds `0.0.0.0` and defaults to **port 8000**. Override it with `--port`
(a **top-level** flag — it must come *before* `policy:checkpoint`). Use the same number
on the tunnel and on `--remote_port`. The example below uses **8001**.

### Terminal 1 — GPU box (`arthur@40.78.176.163`, openpi repo)

```bash
cd /mnt/data3/arthur/openpi
uv run scripts/serve_policy.py --port 8001 policy:checkpoint \
    --policy.config=pi05_umi_rizon10 \
    --policy.dir=checkpoints/pi05_umi_rizon10/my_experiment/29999
```

Leave this running. You should see a log line with the hostname / IP; the websocket
is now listening on port 8001.

### Terminal 2 — robot laptop, SSH tunnel

In a separate terminal on the robot machine, set up local port forwarding. Any traffic
sent to port 8001 on the laptop is forwarded to port 8001 on the GPU box. Keep this
open for the whole rollout (`-N` means "no remote command, just the tunnel"):

```bash
ssh -N -L 8001:localhost:8001 arthur@40.78.176.163
```

If the GPU box already listens on 8000 and you did not pass `--port`, forward 8000 instead:
`ssh -N -L 8000:localhost:8000 arthur@40.78.176.163`.

### Terminal 3 — robot laptop, client (`flexiv_zed` env)

This is **not** the openpi training venv. `main.py` needs `flexivrdk`, `openpi-client`,
`numpy`, `opencv-python`, `tyro`. Put the openpi source tree on `PYTHONPATH` so
`openpi.shared.se3` imports without installing JAX on the robot:

```bash
conda activate flexiv_zed
cd /home/bimanual/arthur/openpi
PYTHONPATH=/home/bimanual/arthur/openpi/src python examples/umi_rizon10/main.py \
    --robot_sn Rizon10-062394 \
    --gripper_name Grav \
    --remote_host localhost \
    --remote_port 8001 \
    --prompt "pick up the tape and place it in the bin"
```

`--remote_host localhost` is required once the tunnel is up; do not point at the GPU
public IP from the client. `main.py` replans every `open_loop_horizon=8` steps (0.4 s)
and paces the loop at 20 Hz.

### Before the first rollout

- **Wrist camera.** `WristCamera` in `main.py` assumes a plain V4L2 device. Replace it with the
real driver, and make sure it is the *same* camera `T_cam_wrt_tcp` was measured for.
- **Speed.** The human demonstrations reach ~0.6 m/s and ~70 °/s at the 90th percentile
(max ~0.87 m/s, ~138 °/s). The Flexiv NRT defaults are 0.5 m/s and 1.0 rad/s (~57 °/s), so
the robot will lag behind fast chunks. `main.py` starts *below* those defaults
(`--max_linear_vel 0.25`, `--max_angular_vel 0.8`); raise them deliberately.
- `--max_jump_m`**.** Aborts if a commanded pose is more than 30 cm from the current TCP. A
wrong `T_base_wrt_otworld` presents exactly as an immediate large jump, so leave this on for
the first rollouts.



## Known experiments to try

- **Camera slot.** The single UMI camera goes in `left_wrist_0_rgb`, with `base_0_rgb` and
`right_wrist_0_rgb` zero-filled and masked off. π0.5 pretraining rarely masks `base_0_rgb`,
so an all-zero masked base view is somewhat off-distribution. The one-line fallback in
`src/openpi/policies/rizon10_policy.py` is to also put the frame in `base_0_rgb` with its
mask set to `np.True_`.
- **Interpolation** instead of zero-order hold for `state` too (`actions` already interpolate).
- **Absolute actions** in the base frame instead of chunk-relative.
- **Rotation vectors** instead of the 6-D representation (letting the model emit magnitudes
past 2π to dodge the wrapping discontinuity).
- `<control_mode>` prompt tags — not used here; see
[openpi#695](https://github.com/Physical-Intelligence/openpi/issues/695).



## Tests

```bash
uv run pytest src/openpi/shared/se3_test.py
```

On a machine with ROS on the `PYTHONPATH`, prefix with
`PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — ROS's `launch_testing` pytest plugin
otherwise gets autoloaded and fails to import.