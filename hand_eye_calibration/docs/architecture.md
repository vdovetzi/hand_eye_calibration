# Architecture

`hand_eye_calibration` is split by dependency direction, not by framework
class.  Production code lives in three roots only:

```text
cli ───────────────┐
                   v
app -> lib/ros -> lib/core
  \--------------> lib/io
```

`lib/core` contains the calibration mathematics, target detection and pure
capture rules.  It does not include ROS, Qt or filesystem headers.  `lib/io`
owns the on-disk dataset and YAML contracts.  `lib/ros` converts ROS messages
to the typed structures used by the core.  The command-line tools and the two
application processes only coordinate those reusable components.

## Calibration data flow

1. `CaptureSource` timestamps images and robot poses using message header
   stamps, buffers the poses, and exposes the pose closest to an image.
2. `CapturePolicy` checks the time difference, frames, settling window and
   duplicate-pose thresholds.  It is deterministic and unit-testable without
   ROS.
3. `DatasetRepository` commits the image, the compatible tab-separated
   `poses.csv` row and the timestamp audit row as one transaction.
4. `TargetDetector` produces matched image/object points for chessboard,
   ArUco or ChArUco targets.  `TargetPoseEstimator` applies subpixel refinement,
   PnP and the reprojection threshold.
5. `calibrate()` requires at least 15 samples, checks both rotational span and
   rotation-axis diversity, compares the
   five OpenCV methods when `auto` is selected, removes gross MAD outliers, and
   scores the complete invariant
   `T_robot * T_calibration * T_target_camera`.
6. `CalibrationIO` preserves the legacy `T`, `K` and `D` keys and adds method,
   frame and consistency metadata.

The backend and offline CLI build the same typed `CalibrationSample` vector and
call the same `core::calibrate()` entry point.  Solver behavior therefore cannot
silently drift between online and offline workflows.

## Application processes

`hand_eye_backend` owns all mutation: subscriptions, capture decisions, dataset
transactions and calibration.  Its existing `std_srvs/Trigger` services and
JSON responses remain the process boundary.  Configuration and JSON assembly
are private implementation details of `backend_node.cpp`.

`hand_eye_frontend` is deliberately a separate process.  `FrontendWindow`
owns widgets and workflow, `FrontendBackendClient` owns ROS parameters and
service calls, `CameraFeed` owns preview rendering, and `FrontendState` owns
settings/pattern serialization and backend JSON parsing.  Both processes are
started by `calibrate.launch.py`.

## Extension rules

- Add a solver option to `lib/core/calibration.*`; neither UI nor ROS code
  should implement calibration mathematics.
- Add a target in `lib/core/target.*` and extend the existing `PatternConfig`.
  Create a new file only if the target introduces an independently testable
  responsibility.
- Add a robot message adapter in `lib/ros/capture_source.*`, normalizing it to
  `TimedRobotPose`.  Do not leak the message type into `core` or `io`.
- Change the dataset contract only through `DatasetRepository`, with a resume
  and corruption test for the migration.
- Keep both `*_main.cpp` files lifecycle-only.  UI callbacks must delegate to
  one of the existing frontend responsibilities.

No duplicated installed header tree is maintained.  The package root is the
internal include path until these libraries become a documented public API.
