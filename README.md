# Hand-Eye Calibration

ROS 2 Jazzy workspace repository for synchronized eye-in-hand and eye-to-hand
calibration. The ROS package is in [`hand_eye_calibration/`](hand_eye_calibration/),
and `ros_babel_fish/` supplies runtime robot-message adaptation.

Build and start the complete backend/frontend application from the workspace
root:

```bash
colcon build --packages-select hand_eye_calibration --symlink-install
source install/setup.bash
ros2 launch hand_eye_calibration calibrate.launch.py
```

The refactored package has three production-code roots: `lib/`, `cli/`, and
`app/`. See the [package README](hand_eye_calibration/README.md) for setup,
parameters, dataset compatibility, CLI usage, calibration guarantees, and the
live-hardware checklist. See the
[architecture guide](hand_eye_calibration/docs/architecture.md) for dependency
and extension rules.
