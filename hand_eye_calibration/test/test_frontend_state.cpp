#include "app/frontend_state.hpp"

#include <QSettings>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace hand_eye::app {
namespace {

TEST(FrontendState, SerializesEverySupportedPattern) {
  PatternSettings pattern;
  pattern.type = 1;
  pattern.arucoDictionary = 5;
  pattern.arucoId = 17;
  pattern.arucoMarkerSize = 0.042;
  EXPECT_EQ(FrontendState::serializePattern(pattern), "1 5 17 0.0420");

  pattern.type = 2;
  pattern.chessRows = 7;
  pattern.chessColumns = 10;
  pattern.chessCellSize = 0.025;
  EXPECT_EQ(FrontendState::serializePattern(pattern), "2 7 10 0.0250");

  pattern.type = 3;
  pattern.charucoRows = 8;
  pattern.charucoColumns = 11;
  pattern.charucoCellSize = 0.03;
  pattern.charucoMarkerSize = 0.022;
  pattern.charucoDictionary = 6;
  pattern.charucoId = 4;
  EXPECT_EQ(FrontendState::serializePattern(pattern),
            "3 8 11 0.0300 0.0220 6 4");

  pattern.type = 0;
  EXPECT_TRUE(FrontendState::serializePattern(pattern).isEmpty());
}

TEST(FrontendState, ParsesBackendPatternConfiguration) {
  const auto chessboard = FrontendState::parsePattern("2 6 3 0.028");
  ASSERT_TRUE(chessboard);
  EXPECT_EQ(chessboard->type, 2);
  EXPECT_EQ(chessboard->chessRows, 6);
  EXPECT_EQ(chessboard->chessColumns, 3);
  EXPECT_DOUBLE_EQ(chessboard->chessCellSize, 0.028);

  const auto aruco = FrontendState::parsePattern("1 4 17 0.035");
  ASSERT_TRUE(aruco);
  EXPECT_EQ(aruco->arucoDictionary, 4);
  EXPECT_EQ(aruco->arucoId, 17);
  EXPECT_DOUBLE_EQ(aruco->arucoMarkerSize, 0.035);

  EXPECT_FALSE(FrontendState::parsePattern("2 6 missing 0.028"));
  EXPECT_FALSE(FrontendState::parsePattern(""));
}

TEST(FrontendState, ParsesLegacyAndRobustBackendFields) {
  const QString json = R"({
    "configured":true,
    "collecting":true,
    "dataset_prepared":true,
    "arm_topic":"/arm_pose",
    "image_topic":"/camera/image",
    "camera_info_topic":"/camera/camera_info",
    "dataset_path":"/tmp/dataset",
    "calibration_path":"/tmp/calibration.yaml",
    "pattern_info":"2 6 3 0.028",
    "intrinsics_path":"/tmp/intrinsics.yaml",
    "poses_format":1,
    "eye_to_hand":true,
    "robot_base_frame":"sgr532/base_link",
    "robot_effector_frame":"sgr532/link_tcp",
    "camera_frame":"realsense",
    "target_frame":"calibration_target",
    "target_tf_visible":true,
    "target_reprojection_px":0.278,
    "target_pose_filter_time_constant_s":0.25,
    "target_pose_filter_enabled":true,
    "target_pose_filter_initialized":true,
    "calibration_tf_available":true,
    "camera_tf_source":"online",
    "camera_tf_active":true,
    "camera_tf_parent_frame":"sgr532/base_link",
    "initial_camera_tf_configured":true,
    "gripper_tf_active":true,
    "gripper_visualization_frame":"calibration_gripper",
    "target_markers_topic":"/calibration/target_markers",
    "min_samples":15,
    "progress_percent":100.0,
    "next_sample_index":18,
    "image_count":18,
    "pose_count":18,
    "statistics":{
      "samples":18,
      "valid_pose_rows":18,
      "translation_spread_x":0.1,
      "translation_spread_y":0.2,
      "translation_spread_z":0.3,
      "rotation_spread_deg":46.0,
      "translation_coverage_score":0.8,
      "rotation_coverage_score":1.0
    },
    "last_validation_rms":0.0012,
    "last_validation_rms_status":"ready",
    "calibration_method":"Park",
    "translation_rms_m":0.002,
    "rotation_rms_deg":0.4,
    "rejected_sample_count":2,
    "rejected_sample_indices":[3,11]
  })";

  const auto state = FrontendState::parseBackendState(json);
  ASSERT_TRUE(state);
  EXPECT_TRUE(state->configured);
  EXPECT_EQ(state->cameraInfoTopic, "/camera/camera_info");
  EXPECT_EQ(state->patternInfo, "2 6 3 0.028");
  EXPECT_EQ(state->intrinsicsPath, "/tmp/intrinsics.yaml");
  EXPECT_TRUE(state->eyeToHand);
  EXPECT_EQ(state->robotBaseFrame, "sgr532/base_link");
  EXPECT_EQ(state->robotEffectorFrame, "sgr532/link_tcp");
  EXPECT_EQ(state->cameraFrame, "realsense");
  EXPECT_EQ(state->targetFrame, "calibration_target");
  EXPECT_TRUE(state->targetTfVisible);
  ASSERT_TRUE(state->targetReprojectionPixels);
  EXPECT_DOUBLE_EQ(*state->targetReprojectionPixels, 0.278);
  EXPECT_DOUBLE_EQ(state->targetPoseFilterTimeConstantSeconds, 0.25);
  EXPECT_TRUE(state->targetPoseFilterEnabled);
  EXPECT_TRUE(state->targetPoseFilterInitialized);
  EXPECT_TRUE(state->calibrationTfAvailable);
  EXPECT_EQ(state->cameraTfSource, "online");
  EXPECT_TRUE(state->cameraTfActive);
  EXPECT_EQ(state->cameraTfParentFrame, "sgr532/base_link");
  EXPECT_TRUE(state->initialCameraTfConfigured);
  EXPECT_TRUE(state->gripperTfActive);
  EXPECT_EQ(state->gripperVisualizationFrame, "calibration_gripper");
  EXPECT_EQ(state->targetMarkersTopic, "/calibration/target_markers");
  EXPECT_EQ(state->minimumSamples, 15);
  EXPECT_EQ(state->imageCount, 18);
  ASSERT_TRUE(state->statistics);
  EXPECT_DOUBLE_EQ(state->statistics->rotationSpreadDegrees, 46.0);
  ASSERT_TRUE(state->lastValidationRms);
  EXPECT_DOUBLE_EQ(*state->lastValidationRms, 0.0012);
  EXPECT_EQ(state->calibrationMethod, "Park");
  EXPECT_EQ(state->rejectedSampleCount, 2);
  EXPECT_EQ(state->rejectedSampleIndices, (std::vector<int>{3, 11}));

  FrontendSettings settings;
  settings.validationRmsStatistics = true;
  const FrontendViewState view = FrontendState::makeViewState(*state, settings);
  EXPECT_TRUE(view.calibrationReady);
  EXPECT_EQ(view.progressValue, 100);
  EXPECT_TRUE(view.statisticsLines.contains("Calibration method: Park"));
  EXPECT_TRUE(view.statisticsLines.contains("Rejected samples: 2"));
  EXPECT_TRUE(view.statisticsLines.contains("Rejected sample indices: 3, 11"));
  EXPECT_TRUE(view.statisticsLines.contains(
      "Live target TF: visible (0.278 px reprojection)"));
  EXPECT_TRUE(
      view.statisticsLines.contains("Marker smoothing: 0.25 s (active)"));
  EXPECT_TRUE(
      view.statisticsLines.contains("Live camera TF: online estimate active"));
  EXPECT_TRUE(view.statisticsLines.contains(
      "Live gripper TF: visible (calibration_gripper)"));
}

TEST(FrontendState, ReportsInitialCameraTransformAndGripperWaitingState) {
  BackendState state;
  state.cameraTfSource = "initial";
  state.cameraTfActive = true;
  state.initialCameraTfConfigured = true;
  state.cameraTfParentFrame = "sgr532/base_link";
  state.cameraFrame = "realsense";
  state.gripperVisualizationFrame = "hand_eye_gripper_pose";

  FrontendSettings settings;
  const FrontendViewState view = FrontendState::makeViewState(state, settings);
  EXPECT_TRUE(view.statisticsLines.contains(
      "Live camera TF: initial guess active (sgr532/base_link -> realsense)"));
  EXPECT_TRUE(view.statisticsLines.contains(
      "Live gripper TF: waiting for gripper pose"));
}

TEST(FrontendState, SuppliesSafeDefaultsForLegacyBackendState) {
  const auto online = FrontendState::parseBackendState(R"({
    "eye_to_hand":true,
    "robot_base_frame":"base_link",
    "camera_frame":"camera",
    "calibration_tf_available":true
  })");
  ASSERT_TRUE(online);
  EXPECT_EQ(online->cameraTfSource, "online");
  EXPECT_TRUE(online->cameraTfActive);
  EXPECT_EQ(online->cameraTfParentFrame, "base_link");
  EXPECT_FALSE(online->initialCameraTfConfigured);
  EXPECT_FALSE(online->gripperTfActive);
  EXPECT_EQ(online->gripperVisualizationFrame, "hand_eye_gripper_pose");
  EXPECT_EQ(online->targetMarkersTopic, "/hand_eye_backend/target_markers");
  EXPECT_DOUBLE_EQ(online->targetPoseFilterTimeConstantSeconds, 0.0);
  EXPECT_FALSE(online->targetPoseFilterEnabled);
  EXPECT_FALSE(online->targetPoseFilterInitialized);

  const auto malformedOptionalFields = FrontendState::parseBackendState(R"({
    "camera_tf_source":"unexpected",
    "camera_tf_active":"true",
    "initial_camera_tf_configured":3,
    "gripper_tf_active":"true",
    "gripper_visualization_frame":42,
    "target_markers_topic":""
  })");
  ASSERT_TRUE(malformedOptionalFields);
  EXPECT_EQ(malformedOptionalFields->cameraTfSource, "none");
  EXPECT_FALSE(malformedOptionalFields->cameraTfActive);
  EXPECT_FALSE(malformedOptionalFields->initialCameraTfConfigured);
  EXPECT_FALSE(malformedOptionalFields->gripperTfActive);
  EXPECT_EQ(malformedOptionalFields->gripperVisualizationFrame,
            "hand_eye_gripper_pose");
  EXPECT_EQ(malformedOptionalFields->targetMarkersTopic,
            "/hand_eye_backend/target_markers");
}

TEST(FrontendState, RejectsMalformedBackendState) {
  EXPECT_FALSE(FrontendState::parseBackendState("not json"));
  EXPECT_FALSE(FrontendState::parseBackendState("[]"));
}

TEST(FrontendState, PreservesExistingSettingsKeys) {
  QTemporaryDir directory;
  ASSERT_TRUE(directory.isValid());
  QSettings storage(directory.filePath("settings.ini"), QSettings::IniFormat);

  FrontendSettings input;
  input.armTopic = "/robot/pose";
  input.imageTopic = "/camera/image";
  input.datasetPath = "/tmp/hand_eye";
  input.calibrationPath = "/tmp/result.yaml";
  input.intrinsicsPath = "/tmp/camera.yaml";
  input.pattern.type = 3;
  input.pattern.charucoRows = 9;
  input.pattern.charucoColumns = 12;
  input.pattern.charucoCellSize = 0.03;
  input.pattern.charucoMarkerSize = 0.02;
  input.pattern.charucoDictionary = 5;
  input.pattern.charucoId = 7;
  input.posesFormat = 4;
  input.eyeToHand = true;
  input.validationRmsStatistics = true;

  FrontendState::save(storage, input);
  const auto output = FrontendState::load(storage);
  ASSERT_TRUE(output);
  EXPECT_EQ(output->armTopic, input.armTopic);
  EXPECT_EQ(output->imageTopic, input.imageTopic);
  EXPECT_EQ(output->calibrationPath, input.calibrationPath);
  EXPECT_EQ(output->pattern.type, input.pattern.type);
  EXPECT_EQ(output->pattern.charucoRows, input.pattern.charucoRows);
  EXPECT_DOUBLE_EQ(output->pattern.charucoMarkerSize,
                   input.pattern.charucoMarkerSize);
  EXPECT_EQ(output->posesFormat, input.posesFormat);
  EXPECT_TRUE(output->eyeToHand);
  EXPECT_TRUE(output->validationRmsStatistics);
}

} // namespace
} // namespace hand_eye::app
