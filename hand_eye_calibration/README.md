# Hand-Eye Calibration

ROS 2 package for collecting calibration datasets and computing hand-eye
calibration from robot poses and camera images.

The package keeps the original CLI tools, and also adds a backend/frontend app
for a guided workflow: choose topics, collect samples, inspect dataset coverage,
preview detected calibration pattern points, remove bad samples, and run
calibration.

![Settings screen](resources/readme/settings.png)

![Dataset workflow](resources/readme/workflow.png)

## Features

- ROS 2 Jazzy C++ package.
- CLI tools for dataset collection, calibration, and validation.
- `hand_eye_backend` ROS node with services for a combined application flow.
- Qt `hand_eye_frontend` app built from a `.ui` file.
- Camera preview with detected pattern overlay.
- Dataset resume, overwrite, and last-sample removal.
- Live dataset statistics:
  - translation coverage;
  - rotation coverage;
  - Target lock RMS.
- Support for raw `sensor_msgs/msg/Image` and compressed image transport.
- Arm messages are read through `ros_babel_fish`, so the backend can inspect
  topic types at runtime.

## Build

From your ROS 2 workspace:

```bash
cd ~/hand_eye_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select hand_eye_calibration --symlink-install
source install/setup.bash
```

For development builds:

```bash
colcon build \
  --packages-select hand_eye_calibration \
  --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Sanitizers are enabled by default in Debug builds and can be controlled with:

```bash
colcon build \
  --packages-select hand_eye_calibration \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZERS=ON
```

## App Workflow

Start the backend:

```bash
ros2 run hand_eye_calibration hand_eye_backend
```

Start the frontend:

```bash
ros2 run hand_eye_calibration hand_eye_frontend
```

The app opens on the Settings screen.

Required settings:

- **Arm topic**: robot/gripper pose source.
- **Camera topic**: camera image source.
- **Dataset folder**: output folder for captured images and `poses.csv`.
- **Calibration path**: output YAML path, defaulting to `calibration.yaml`.
- **Pose format**: how each row in `poses.csv` is interpreted.
- **Pattern type**: ArUco, chessboard, or ChArUco.

Optional settings:

- **Intrinsics YAML**: camera intrinsics. If empty, intrinsics are estimated from
  the dataset where possible.
- **Eye to hand**: enable for eye-to-hand setups; disable for eye-in-hand.

After pressing **Next**, the workflow screen shows capture controls, the camera
stream, detected pattern points, selected statistics, and the log panel.

Use:

- **Capture** to save the current image and arm pose.
- **Remove sample** to delete the latest saved image and pose row.
- **Calibrate** to compute and save the hand-eye transform.

If the dataset folder already exists, the app asks whether to continue the
dataset or start over.

## Dataset Layout

Captured datasets use this structure:

```text
dataset/
  images/
    0.png
    1.png
    2.png
  poses.csv
```

`poses.csv` is written as tab-separated values. For pose-space calibration,
supported pose formats are:

```text
1: x y z qx qy qz qw
2: x y z qw qx qy qz
3: x y z roll pitch yaw [rad]
4: x y z roll pitch yaw [deg]
5: x y z yaw pitch roll [rad]
6: x y z yaw pitch roll [deg]
7: joint positions j0 j1 ... jn
```

Joint-state calibration is recognized in the UI, but pose-space calibration
statistics and calibration from joints are not implemented yet.

## Pattern Parameters

The app exposes pattern-specific fields. The CLI tools use the compact
`pattern_info` format:

```text
ArUco:      "1 <dictionary> <marker_id> <marker_size_m>"
Chessboard: "2 <rows> <cols> <cell_size_m>"
ChArUco:    "3 <rows> <cols> <cell_size_m> <marker_size_m> <dictionary> <marker_id>"
```

Examples:

```bash
--pattern-type "2 6 9 0.025"
--pattern-type "1 4 12 0.04"
```

For chessboards, `rows` and `cols` are the number of inner corners.

## Dataset Statistics

The frontend shows only the statistics selected on the Settings screen.

### Translation Coverage

Translation coverage measures how well saved gripper positions cover translation
space. It is not just the max-min span: the score also accounts for the number
of samples. With only one or two samples, coverage is zero even if those samples
are far apart.

The UI also shows the raw xyz span in meters for debugging.

### Rotation Coverage

Rotation coverage measures orientation diversity around the gripper. Like
translation coverage, it is down-weighted when there are too few samples, so two
extreme poses do not look like a dense dataset.

The UI also shows the raw maximum rotation spread in degrees.

### Target Lock RMS

Target lock RMS estimates how much the calibration target origin moves in the
gripper frame. If the target is rigidly attached to the gripper, this value
should be close to zero.

The backend updates it after each captured sample once there are enough samples
for a quick calibration.

## CLI Tools

The separate nodes are still available.

Collect a dataset:

```bash
ros2 run hand_eye_calibration dataset_collector \
  --arm-topic "/gripper/pose" \
  --image-topic "/camera/color/image_raw" \
  --dataset-path "dataset" \
  --pattern-type "2 6 9 0.025"
```

Compute calibration:

```bash
ros2 run hand_eye_calibration compute_calibration \
  --dataset-path "dataset" \
  --poses-format 1 \
  --pattern-type "2 6 9 0.025"
```

Validate calibration:

```bash
ros2 run hand_eye_calibration validate_calibration \
  --dataset-path "dataset" \
  --calibration-path "calibration.yaml" \
  --pattern-type "2 6 9 0.025"
```

## Backend Services

`hand_eye_backend` exposes `std_srvs/srv/Trigger` services:

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

Configuration is passed through ROS parameters before service calls. The
frontend handles this automatically.

## Troubleshooting

### `tf2/LinearMath/Matrix3x3.h` not found

Make sure the ROS environment is sourced and dependencies are installed:

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

For IDEs and clangd, build with compile commands:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### OpenCV linker warnings from `cv_bridge`

If you see warnings about ROS OpenCV libraries conflicting with local OpenCV
libraries, check that your environment is not mixing incompatible OpenCV
installations. The package may still build, but runtime behavior should be
tested carefully.

### Pattern is not detected

Check:

- camera topic is correct;
- image is not blurred or overexposed;
- pattern dimensions match the real board;
- chessboard `rows` and `cols` are inner corners;
- marker/cell sizes are given in meters.

## Credits

Made by [vdovetzi](https://github.com/vdovetzi) with love to robotics and the
community.
