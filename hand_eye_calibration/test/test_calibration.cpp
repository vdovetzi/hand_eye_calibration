#include "lib/core/calibration.hpp"

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace hand_eye::core {
namespace {

RigidTransform transform(const cv::Vec3d &rotationVector,
                         const cv::Vec3d &translation) {
  cv::Mat rotation;
  cv::Rodrigues(cv::Mat(rotationVector), rotation);
  return {rotation, cv::Mat(translation)};
}

double rotationErrorDegrees(const RigidTransform &left,
                            const RigidTransform &right) {
  const cv::Mat relative = left.rotation.t() * right.rotation;
  const double cosine =
      std::clamp((cv::trace(relative)[0] - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cosine) * 180.0 / CV_PI;
}

std::vector<RigidTransform> robotPoses(const std::size_t count) {
  std::vector<RigidTransform> poses;
  poses.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const double value = static_cast<double>(index);
    poses.push_back(
        transform({0.055 * value, 0.24 * std::sin(0.63 * value),
                   0.19 * std::cos(0.41 * value)},
                  {0.24 + 0.012 * value, -0.14 + 0.05 * std::sin(0.5 * value),
                   0.35 + 0.025 * std::cos(0.31 * value)}));
  }
  return poses;
}

std::vector<CalibrationSample>
eyeInHandSamples(const std::size_t count, const RigidTransform &calibration) {
  const RigidTransform fixedTarget =
      transform({-0.1, 0.25, 0.08}, {0.55, -0.12, 0.42});
  std::vector<CalibrationSample> samples;
  for (const RigidTransform &robot : robotPoses(count)) {
    samples.push_back(
        {robot, calibration.inverse() * robot.inverse() * fixedTarget});
  }
  return samples;
}

std::vector<CalibrationSample>
eyeToHandSamples(const std::size_t count, const RigidTransform &calibration) {
  const RigidTransform fixedTargetInGripper =
      transform({0.04, -0.08, 0.02}, {0.02, 0.01, 0.15});
  std::vector<CalibrationSample> samples;
  for (const RigidTransform &robot : robotPoses(count)) {
    samples.push_back(
        {robot, calibration.inverse() * robot * fixedTargetInGripper});
  }
  return samples;
}

void expectNear(const RigidTransform &actual, const RigidTransform &expected,
                const double tolerance) {
  EXPECT_LT(cv::norm(actual.translation - expected.translation), tolerance);
  EXPECT_LT(rotationErrorDegrees(actual, expected), tolerance * 100.0);
}

TEST(RigidTransformTest, MatrixCompositionAndInverseAreConsistent) {
  const RigidTransform value = transform({0.2, -0.1, 0.3}, {0.4, -0.2, 0.7});
  const RigidTransform restored = RigidTransform::fromMatrix(value.matrix());

  expectNear(restored, value, 2e-8);
  expectNear(value * value.inverse(), RigidTransform::identity(), 2e-8);
}

TEST(CalibrationTest, AutoRecoversEyeInHandTransform) {
  const RigidTransform expected =
      transform({0.18, -0.12, 0.09}, {0.045, -0.025, 0.082});
  CalibrationOptions options;
  options.mode = CalibrationMode::EYE_IN_HAND;
  const CalibrationResult result =
      calibrate(eyeInHandSamples(20, expected), options);

  EXPECT_NE(result.method, CalibrationMethod::AUTO);
  EXPECT_EQ(result.usedSampleIndices.size(), 20U);
  EXPECT_TRUE(result.rejectedSampleIndices.empty());
  EXPECT_LT(result.consistency.translationRmsMeters, 1e-7);
  EXPECT_LT(result.consistency.rotationRmsDegrees, 1e-5);
  expectNear(result.transform, expected, 1e-6);
}

TEST(CalibrationTest, AutoRecoversEyeToHandTransform) {
  const RigidTransform expected =
      transform({-0.12, 0.06, 0.16}, {0.31, -0.18, 0.52});
  CalibrationOptions options;
  options.mode = CalibrationMode::EYE_TO_HAND;
  const CalibrationResult result =
      calibrate(eyeToHandSamples(20, expected), options);

  EXPECT_NE(result.method, CalibrationMethod::AUTO);
  EXPECT_LT(result.consistency.translationRmsMeters, 1e-7);
  EXPECT_LT(result.consistency.rotationRmsDegrees, 1e-5);
  expectNear(result.transform, expected, 1e-6);
}

TEST(CalibrationTest, EveryOpenCvMethodRecoversSyntheticTransform) {
  const RigidTransform expected =
      transform({0.18, -0.12, 0.09}, {0.045, -0.025, 0.082});
  const std::vector<CalibrationMethod> methods{
      CalibrationMethod::TSAI, CalibrationMethod::PARK,
      CalibrationMethod::HORAUD, CalibrationMethod::ANDREFF,
      CalibrationMethod::DANIILIDIS};

  for (const CalibrationMethod method : methods) {
    CalibrationOptions options;
    options.mode = CalibrationMode::EYE_IN_HAND;
    options.method = method;
    const CalibrationResult result =
        calibrate(eyeInHandSamples(20, expected), options);

    EXPECT_EQ(result.method, method);
    expectNear(result.transform, expected, 1e-5);
  }
}

TEST(CalibrationTest, RejectsUnobservableRobotMotion) {
  std::vector<CalibrationSample> samples(15);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    samples[index].robotPose.translation.at<double>(0) =
        0.01 * static_cast<double>(index);
  }

  EXPECT_THROW(calibrate(samples), std::invalid_argument);
}

TEST(CalibrationTest, EnforcesFifteenSampleSafetyFloor) {
  const RigidTransform expected =
      transform({0.18, -0.12, 0.09}, {0.045, -0.025, 0.082});
  CalibrationOptions options;
  options.minimumSamples = kMinimumCalibrationSamples - 1;

  EXPECT_THROW(calibrate(eyeInHandSamples(20, expected), options),
               std::invalid_argument);
}

TEST(CalibrationTest, RejectsLargeRotationAroundOnlyOneAxis) {
  std::vector<CalibrationSample> samples(15);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const double angle = 0.09 * static_cast<double>(index);
    samples[index].robotPose =
        transform({0.0, 0.0, angle}, {0.01 * index, 0.0, 0.0});
  }

  EXPECT_GT(rotationSpanDegrees(samples), 35.0);
  EXPECT_THROW(calibrate(samples), std::invalid_argument);
}

TEST(CalibrationTest, RemovesGrossOutlierAndRetainsExactSolution) {
  const RigidTransform expected =
      transform({0.18, -0.12, 0.09}, {0.045, -0.025, 0.082});
  std::vector<CalibrationSample> samples = eyeInHandSamples(21, expected);
  samples.back().targetToCamera.translation.at<double>(0) += 0.35;
  samples.back().targetToCamera.translation.at<double>(2) -= 0.20;

  const CalibrationResult result = calibrate(samples);

  ASSERT_EQ(result.rejectedSampleIndices.size(), 1U);
  EXPECT_EQ(result.rejectedSampleIndices.front(), 20U);
  EXPECT_EQ(result.usedSampleIndices.size(), 20U);
  expectNear(result.transform, expected, 1e-6);
}

TEST(CalibrationTest, FullInvariantDetectsRobotAndTargetInconsistency) {
  const RigidTransform expected =
      transform({0.18, -0.12, 0.09}, {0.045, -0.025, 0.082});
  std::vector<CalibrationSample> samples = eyeInHandSamples(15, expected);
  samples[4].robotPose.translation.at<double>(1) += 0.03;

  const ConsistencyMetrics metrics =
      evaluateConsistency(samples, expected, CalibrationMode::EYE_IN_HAND);
  EXPECT_GT(metrics.translationRmsMeters, 0.005);
  EXPECT_GT(metrics.translationMaxMeters, 0.02);
}

} // namespace
} // namespace hand_eye::core
