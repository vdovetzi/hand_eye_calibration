# Hand-Eye Calibration

ROS 2 Jazzy package for synchronized dataset collection, robust eye-in-hand or
eye-to-hand calibration, and live validation. The backend and Qt frontend are
separate processes, but the complete application starts with one command:

```bash
ros2 launch hand_eye_calibration calibrate.launch.py
```

The default launch loads `config/default.yaml`. Use another parameter file with:

```bash
ros2 launch hand_eye_calibration calibrate.launch.py \
  config:=/absolute/path/to/calibration.yaml
```

## Build

From the ROS 2 workspace root:

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select hand_eye_calibration --symlink-install
source install/setup.bash
```

The package uses the ROS-compatible OpenCV installation, OpenCV contrib for
ArUco/ChArUco, Boost, `ros_babel_fish`, Qt 5 Widgets, and the ROS 2 Jazzy RViz
libraries. Jazzy's RViz is Qt 5-based, so the frontend must use Qt 5 as well.

## Design

Production code has three roots:

```text
lib/core   calibration, target detection, and pure capture rules
lib/io     transactional datasets and calibration YAML
lib/ros    ROS message normalization and timestamped pose buffering
cli        collection, offline calibration, and validation commands
app        the backend process and the Qt frontend process
```

`lib/core` has no ROS, Qt, or filesystem dependency. Both the backend and
`compute_calibration` use the same typed dataset/calibration pipeline. See
`docs/architecture.md` for dependency and extension rules.

## Application workflow

1. Set the arm pose topic, image topic, target, dataset path, and calibration
   mode in the frontend or parameter YAML.
2. Configure and prepare the dataset. Existing data must be explicitly resumed
   or overwritten.
3. The workflow opens on **Camera + Detection**. Verify the live target
   overlay, start collection, wait for the robot to settle, and capture varied
   poses.
4. Run calibration after at least 15 accepted observations with at least 35
   degrees of rotational excitation.
5. Open **Live 3D Calibration** to inspect the robot, camera, gripper pose, and
   detected target while samples are collected. Use the expand control or F11
   for the full interactive RViz view.
6. Inspect the selected method, rejected samples, and consistency RMS values,
   then validate on the live robot.

The image and closest robot pose are matched by their source timestamps. A
capture is rejected when timestamps, configured frames, CameraInfo resolution,
stationarity, target detection, or pose diversity checks fail.

## Backend compatibility

The executables remain:

- `hand_eye_backend`
- `hand_eye_frontend`

The backend keeps the existing private `std_srvs/srv/Trigger` services:

```text
/hand_eye_backend/configure
/hand_eye_backend/get_state
/hand_eye_backend/prepare_dataset
/hand_eye_backend/start_collection
/hand_eye_backend/capture_sample
/hand_eye_backend/remove_sample
/hand_eye_backend/stop_collection
/hand_eye_backend/run_calibration
```

No custom ROS messages are required. `get_state` returns the compatible JSON
state in `response.message`; the command services retain normal Trigger
success/error messages. The JSON keeps the existing workflow and statistics
fields and may additionally report `calibration_method`, `translation_rms_m`,
`rotation_rms_deg`, `rejected_sample_count`, `rejected_sample_indices`, the
configured robot/camera/target frames, `target_tf_visible`,
`target_reprojection_px`, and `calibration_tf_available`.

## Parameters

These parameters are loaded by `config/default.yaml`. Empty topic and frame
values must be filled for a live setup. Empty expected frame values disable the
corresponding name check.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `arm_topic` | `""` | Cartesian robot pose/transform topic |
| `image_topic` | `""` | `Image` or `CompressedImage` topic |
| `camera_info_topic` | `""` | Optional `CameraInfo` topic |
| `robot_base_frame` | `""` | Expected robot pose parent frame |
| `robot_effector_frame` | `""` | Expected robot pose child frame |
| `camera_frame` | `""` | Expected image/CameraInfo frame |
| `target_frame` | `""` | Live detected target child frame; derived from target type when empty |
| `gripper_visualization_frame` | `hand_eye_gripper_pose` | TF alias driven by the configured robot pose topic; avoids publishing over the URDF TCP frame |
| `dataset_path` | `dataset` | Dataset directory |
| `calibration_path` | `calibration.yaml` | Output YAML file or directory |
| `pattern_info` | `""` | Compact target description shown below |
| `poses_format` | `1` | `poses.csv` representation, 1 through 6 |
| `eye_to_hand` | `true` | Fixed camera when true; camera on effector when false |
| `intrinsics_path` | `""` | YAML containing `K` and `D` |
| `calibration_method` | `auto` | `auto`, `tsai`, `park`, `horaud`, `andreff`, or `daniilidis` |
| `require_pattern_detection` | `true` | Detect the configured target before saving |
| `require_stamped_samples` | `true` | Reject messages without source timestamps |
| `publish_live_tf` | `true` | Publish live target and online calibration transforms |
| `publish_gripper_tf` | `true` | Publish the calibration-only gripper pose alias |
| `target_pose_filter_time_constant_s` | `0.25` | Time-aware target TF smoothing; `0` disables it, larger values smooth more strongly |
| `initial_camera_x` | `1.0` | Initial parent-to-camera X translation in metres |
| `initial_camera_y` | `0.0` | Initial parent-to-camera Y translation in metres |
| `initial_camera_z` | `0.0` | Initial parent-to-camera Z translation in metres |
| `initial_camera_qx` | `0.0` | Initial parent-to-camera quaternion X |
| `initial_camera_qy` | `0.0` | Initial parent-to-camera quaternion Y |
| `initial_camera_qz` | `0.0` | Initial parent-to-camera quaternion Z |
| `initial_camera_qw` | `1.0` | Initial parent-to-camera quaternion W |
| `overwrite_dataset` | `false` | Replace an existing dataset on prepare |
| `resume_dataset` | `true` | Resume a valid existing dataset |
| `min_samples` | `15` | Minimum observations passed to the solver; values below the hard floor of 15 are clamped/rejected |
| `min_rotation_span_deg` | `35.0` | Minimum robot rotational excitation |
| `min_rotation_axis_separation_deg` | `10.0` | Minimum diversity between robot rotation axes |
| `max_reprojection_px` | `1.0` | Maximum per-view target reprojection RMS |
| `max_time_delta_ms` | `30.0` | Maximum image/robot timestamp difference |
| `settle_window_ms` | `500.0` | Required stationary history before capture |
| `stationary_translation_m` | `0.0005` | Maximum translation during settling |
| `stationary_rotation_deg` | `0.25` | Maximum rotation during settling |
| `duplicate_translation_m` | `0.005` | Near-duplicate translation threshold |
| `duplicate_rotation_deg` | `5.0` | Near-duplicate rotation threshold |
| `stat_translation_spread` | `true` | Show translation coverage statistics |
| `stat_rotation_spread` | `true` | Show rotation coverage statistics |
| `stat_validation_rms` | `true` | Show target-lock/validation statistics |

Set `require_stamped_samples: false` only as a compatibility fallback for
unstamped `Pose` or `Transform` messages. They use receipt time and therefore
provide weaker synchronization than `PoseStamped` or `TransformStamped`.

## Targets and camera intrinsics

`pattern_info` and the CLI target options use these strings (all sizes are in
metres):

```text
ArUco:      "1 <dictionary_bits:4..7> <marker_id> <marker_size>"
Chessboard: "2 <inner_corner_rows> <inner_corner_columns> <cell_size>"
ChArUco:    "3 <square_rows> <square_columns> <square_size> <marker_size> <dictionary_bits:4..7>"
```

Examples:

```text
1 4 12 0.040
2 6 9 0.025
3 7 5 0.040 0.020 4
```

Chessboard, ArUco, and ChArUco observations use subpixel corner refinement.
Known intrinsics can come from `CameraInfo` during collection or from a YAML
file containing `K` and `D`. Both OpenCV matrix YAML and ROS parameter YAML such
as `/**/ros__parameters/K,D` are accepted. Chessboard and ChArUco datasets can
estimate intrinsics from multiple views when none are supplied. A single ArUco
marker cannot safely determine camera intrinsics, so ArUco calibration fails
instead of silently inventing them.

## Sagittarius live session

From the workspace root, build and source the packages, then run the complete
arm, RealSense, robot-description, backend, and frontend session:

```bash
source /opt/ros/jazzy/setup.zsh
colcon build --packages-up-to sagittarius_perception --symlink-install
source install/setup.zsh
ros2 launch sagittarius_perception hand_eye_calibration_session.launch.py
```

The session defaults to `~/.ros2/realsense_intrinsics_960x540_ros.yaml`, a
960 x 540 color stream, `/gripper/pose`, and target description
`pattern_info:="2 6 3 0.028"` (a 6 x 3 chessboard with 0.028 m squares).
Override a value with a launch argument when needed, for example:

```bash
ros2 launch sagittarius_perception hand_eye_calibration_session.launch.py \
  intrinsics_file:=/absolute/path/to/realsense_intrinsics.yaml
```

To calibrate with an 81 mm ArUco marker from the 4x4 dictionary using ID 1,
start the same session with:

```bash
ros2 launch sagittarius_perception hand_eye_calibration_session.launch.py \
  open_live_3d:=true \
  pattern_info:="1 4 1 0.081"
```

The four fields are target type, dictionary bit size, marker ID, and measured
outer marker size in metres. The live view renders the canonical marker cells
and labels them with the configured dictionary and ID, so an incorrect target
selection is visible immediately. ArUco requires valid known intrinsics; the
session passes the RealSense YAML to the backend by default.

Set the rough parent-to-camera pose on the command line when the default
translation `(1, 0, 0)` and identity quaternion are not useful:

```bash
ros2 launch sagittarius_perception hand_eye_calibration_session.launch.py \
  initial_camera_x:=0.42 initial_camera_y:=-0.18 initial_camera_z:=0.55 \
  initial_camera_qx:=0.0 initial_camera_qy:=0.0 \
  initial_camera_qz:=0.7071068 initial_camera_qw:=0.7071068
```

For eye-to-hand this is `base_frame -> camera_frame`; for eye-in-hand it is
`effector_frame -> camera_frame`. The initial transform is available in RViz
before a solution exists, then the backend replaces it with each valid online
calibration result while collection proceeds.

For a direct live-view check, add `open_live_3d:=true`; this enters the workflow
without starting collection, but still opens **Camera + Detection** first.
Embedded RViz is initialized only after selecting **Live 3D Calibration**. Use
its Move Camera, Interact, Select, and Focus tools normally, and use the expand
control or F11 to enlarge the view.

Click **Next** to prepare/start collection, then select **Live 3D Calibration**.
The robot model follows `/joint_states` through `robot_state_publisher`, but
the view does not draw every URDF TF axis. Instead it shows the camera, target,
and `hand_eye_gripper_pose` axes. The latter is a separate visualization alias
fed directly from `/gripper/pose`, so it does not conflict with the fixed
`sgr532/link_tcp` transform owned by `robot_state_publisher`.

While the board is visible, the backend publishes
`camera_frame -> target_frame` at the image timestamp. After at least 15
sufficiently varied accepted samples produce a valid online solution, the
camera axes update from the initial estimate to `base -> camera` for
eye-to-hand or `effector -> camera` for eye-in-hand. The status list identifies
the active camera source and reports target visibility and reprojection error.

The displayed target pose is smoothed with a frame-rate-independent 0.25 s
low-pass filter. It only affects the RViz TF; detection, saved samples, PnP and
calibration continue to use raw measurements. Change the strength while the
session is running (no restart or reconfigure is needed):

```bash
ros2 param set /hand_eye_backend target_pose_filter_time_constant_s 0.5
```

Use `0.0` for the raw pose, or a larger value such as `0.5` for stronger but
slower smoothing.

## Dataset format

```text
dataset/
  images/
    0.png
    1.png
  poses.csv
  samples.tsv
```

`poses.csv` remains tab-separated for compatibility. `samples.tsv` stores the
image and robot source timestamps used for capture auditing. Images, pose rows,
and audit rows are committed transactionally, and sample numbers remain
contiguous when the latest sample is removed.

Every pose begins with translation `x y z` in metres:

| Format | Remaining columns |
| ---: | --- |
| 1 | `qx qy qz qw` |
| 2 | `qw qx qy qz` |
| 3 | `roll pitch yaw` in radians |
| 4 | `roll pitch yaw` in degrees |
| 5 | `yaw pitch roll` in radians |
| 6 | `yaw pitch roll` in degrees |

The live robot topic must represent the Cartesian transform from the configured
base frame to the effector frame. Joint positions are not a pose format.

## CLI

Collect a synchronized dataset. `--output` is a parent directory; the command
creates or resumes its `dataset/` child:

```bash
ros2 run hand_eye_calibration dataset_collector \
  --arm-topic /robot/tool_pose \
  --image-topic /camera/image_raw \
  --camera-info-topic /camera/camera_info \
  --output /tmp/hand_eye_session \
  --pattern-type "2 6 9 0.025" \
  --poses-format 1 \
  --base-frame base_link \
  --effector-frame tool0 \
  --camera-frame camera_optical_frame
```

Press Enter to capture and `q` or Escape to finish. The collector exposes the
same synchronization, settling, frame, and duplicate thresholds as long-form
CLI options; run it with `--help` for the complete list. For an ArUco dataset,
pass `--intrinsics-path`; unlike the backend's live workflow, the standalone
collector does not embed transient `CameraInfo` in the compatible dataset.

Compute an offline calibration through the same core solver as the backend:

```bash
ros2 run hand_eye_calibration compute_calibration \
  --dataset-path /tmp/hand_eye_session/dataset \
  --poses-format 1 \
  --calibration-pattern "2 6 9 0.025" \
  --eye-to-hand=false \
  --method auto \
  --output /tmp/hand_eye_session/calibration.yaml
```

For known intrinsics add `--intrinsics-path camera.yaml`. Useful quality options
include `--min-samples`, `--min-rotation-deg`,
`--min-axis-separation-deg`, `--max-reprojection-px`, and
`--no-outlier-rejection`. The solver always enforces at least 15 accepted
samples even if a lower CLI value is supplied.

Validate against live camera and robot streams with the preserved interface:

```bash
ros2 run hand_eye_calibration validate_calibration \
  --arm-topic /robot/tool_pose \
  --image-topic /camera/image_raw \
  --arm-frame base_link \
  --calibration-path /tmp/hand_eye_session \
  --pattern-type "2 6 9 0.025"
```

By default, clicks only publish the camera/output poses used for visualization;
the validator does not connect to or command the arm topic. Arm commands are
available only for a calibration whose YAML explicitly contains
`calibration_type: eye_to_hand`, because only that result maps camera points
directly into the robot base frame. Enable them deliberately and provide a
tool orientation independent of the observed board orientation:

```bash
ros2 run hand_eye_calibration validate_calibration \
  --arm-topic /robot/tool_pose \
  --image-topic /camera/image_raw \
  --arm-frame base_link \
  --calibration-path /tmp/hand_eye_session \
  --pattern-type "2 6 9 0.025" \
  --publish-arm-command \
  --tool-roll-deg 180 \
  --tool-pitch-deg 0 \
  --tool-yaw-deg 0
```

Eye-in-hand and legacy calibration files without type metadata are rejected
when command publishing is requested. `--calibration-path` is the directory
containing `calibration.yaml`. Run any CLI command with `--help` for all
accepted options and compatibility aliases.

## Solver and output

In `auto` mode the core compares OpenCV's Tsai, Park, Horaud, Andreff, and
Daniilidis solvers. It rejects unobservable datasets, filters gross pose
outliers with a robust median-absolute-deviation threshold, and scores the full
rigid-target invariant:

```text
T_robot * T_calibration * T_target_camera
```

The result reports the chosen method, used/rejected sample indices, translation
consistency RMS in metres, and rotation consistency RMS in degrees. The output
YAML preserves `T`, `K`, and `D`, and adds method, calibration mode, frame, and
RMS metadata plus rejected sample indices. `T` is `T_gripper_camera` for
eye-in-hand and `T_base_camera` for eye-to-hand.

Low RMS is evidence of internal consistency, not proof that frame directions,
target dimensions, timestamps, or robot kinematics are correct.

## Live-hardware checklist

- Use stamped camera and robot messages driven by synchronized clocks.
- Verify that the robot pose direction is base-to-effector and that all frame
  IDs match the configured values.
- Verify CameraInfo belongs to the selected image stream and resolution.
- Keep the target rigid, avoid motion blur, and collect poses across multiple
  rotation axes rather than one arc.
- Validate the saved transform against withheld poses before deploying it.

Synthetic and offline tests cannot validate a particular robot's timestamp
source, transform convention, camera driver, or mechanical rigidity. A final
live robot/camera validation is therefore required for every installation.
