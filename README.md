# Hand–Eye Calibration (ROS 2)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=vdovetzi_hand_eye_calibration&metric=alert_status&token=29ad187166a6675ab0c2e984681a5e750e86b7e6)](https://sonarcloud.io/summary/new_code?id=vdovetzi_hand_eye_calibration)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=vdovetzi_hand_eye_calibration&metric=code_smells&token=29ad187166a6675ab0c2e984681a5e750e86b7e6)](https://sonarcloud.io/summary/new_code?id=vdovetzi_hand_eye_calibration)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=vdovetzi_hand_eye_calibration&metric=duplicated_lines_density&token=29ad187166a6675ab0c2e984681a5e750e86b7e6)](https://sonarcloud.io/summary/new_code?id=vdovetzi_hand_eye_calibration)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=vdovetzi_hand_eye_calibration&metric=security_rating&token=29ad187166a6675ab0c2e984681a5e750e86b7e6)](https://sonarcloud.io/summary/new_code?id=vdovetzi_hand_eye_calibration)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=vdovetzi_hand_eye_calibration&metric=sqale_rating&token=29ad187166a6675ab0c2e984681a5e750e86b7e6)](https://sonarcloud.io/summary/new_code?id=vdovetzi_hand_eye_calibration)
![ROS2](https://img.shields.io/badge/ROS2-Jazzy-orange)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)


Minimal ROS 2 package for **hand–eye calibration** using OpenCV.  
Includes tools for **dataset collection** and **offline calibration**.

---

## Package Contents

This package provides **two executable nodes**:

- `dataset_collector` — interactive dataset recording
- `calibrator` — offline hand–eye calibration

---

## Build

```bash
colcon build --packages-select hand_eye_calibration
source install/setup.bash
```

Requirements:
- ROS 2
- C++20
- OpenCV
- Boost.ProgramOptions

---

## Node: `dataset_collector`

Interactive tool to collect **synchronized robot poses and camera images**.

### Description

- Subscribes to:
  - robot pose topic (arbitrary message type via `ros_babel_fish` package)
  - camera image topic
- Displays live camera stream
- Saves **one pose + one image** per user action
- Validates dataset consistency on exit

### Controls

| Key | Action |
|---|---|
| `Enter` | Save pose + image |
| `Q` | Finish collection |

### Output Structure

```
dataset/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── poses.tsv
```

### Usage

```bash
ros2 run hand_eye_calibration dataset_collector \
  --arm-topic /arm/pose \
  --image-topic /camera/image \
  --output /path/to/output \
  --log-level Info
```

### Options

| Option | Description |
|------|-------------|
| `--arm-topic` | Robot pose topic |
| `--image-topic` | Camera image topic |
| `--output` | Output directory (dataset will be created inside) |
| `--log-level` | Info / Debug / Error / Fatal |

---

## Node: `calibrator`

Offline hand–eye calibration using `cv::calibrateHandEye()`.

### Description

- Loads previously collected dataset
- Detects calibration pattern in images
- Computes camera-to-gripper transformation
- Prints result to stdout

### Supported Calibration Modes

- Eye-to-hand
- Eye-in-hand

### Supported Calibration Patterns

| ID | Pattern | Arguments |
|---|--------|-----|
| 1 | ArUco [in future]  |  `"<dict> <id> <size_m>"` |
| 2 | Chessboard  | `"<rows> <cols> <square_size_m>"` |
| 3 | ChArUco [in future] | `"<rows> <cols> <cell_size_m> <marker_size_m> <dict> <id>"` |

### Supported Pose Formats

| ID | Format |
|---|--------|
| 1 | translation (x y z), quaternion (x y z w) |
| 2 | translation (x y z), quaternion (w x y z) |
| 3 | translation (x y z), RPY (rad) |
| 4 | translation (x y z), RPY (deg) |
| 5 | joint values [in future] |

### Usage

```bash
ros2 run hand_eye_calibration calibrator \
  --dataset-path /path/to \
  --calibration-pattern "2 7 6 0.025" \
  --poses-format 3 \
  --eye-to-hand true \
  --log-level Info
```

### Output

Printed to standard output:

- Rotation matrix (camera → gripper)
- Translation vector

---

## Notes
- Logging level affects ROS, OpenCV, and internal helpers

---

## Thanks

Special thanks to the developers of [ros_babel_fish](https://github.com/LOEWE-emergenCITY/ros_babel_fish) for providing one of the most convenient and powerful introspection tools for ROS 2 in C++

## Contribution

Pull requests and issues are welcome. There are several simple TODOs in the source code, and contributions addressing them are especially appreciated.


## License

Apache-2.0
