#include "lib/core/pose_filter.hpp"

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include <cmath>
#include <limits>

namespace hand_eye::core {
namespace {

RigidTransform pose(const double x, const double angleRadians) {
  const cv::Mat rotationVector =
      (cv::Mat_<double>(3, 1) << 0.0, 0.0, angleRadians);
  cv::Mat rotation;
  cv::Rodrigues(rotationVector, rotation);
  const cv::Mat translation = (cv::Mat_<double>(3, 1) << x, 0.0, 0.0);
  return {rotation, translation};
}

double rotationAngle(const RigidTransform &value) {
  cv::Mat rotationVector;
  cv::Rodrigues(value.rotation, rotationVector);
  return cv::norm(rotationVector);
}

RigidTransform dominantAxisBoundaryPose(const double x) {
  constexpr double angle = 2.0 * CV_PI / 3.0 + 1.0e-5;
  cv::Vec3d axis{x, 1.0, 0.2};
  axis /= cv::norm(axis);
  cv::Mat rotation;
  cv::Rodrigues(cv::Mat(axis * angle), rotation);
  return {rotation, cv::Mat::zeros(3, 1, CV_64F)};
}

TEST(PoseFilterTest, RejectsInvalidTimeConstants) {
  EXPECT_THROW(ExponentialPoseFilter(-0.1), std::invalid_argument);
  EXPECT_THROW(ExponentialPoseFilter(std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
}

TEST(PoseFilterTest, RejectsNonFiniteMeasurements) {
  ExponentialPoseFilter filter(0.25);
  RigidTransform invalid = pose(0.0, 0.0);
  invalid.translation.at<double>(0) = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW((void)filter.update(invalid, 1'000'000'000),
               std::invalid_argument);
  EXPECT_FALSE(filter.initialized());
}

TEST(PoseFilterTest, ZeroTimeConstantPassesMeasurementsThrough) {
  ExponentialPoseFilter filter(0.0);
  (void)filter.update(pose(0.0, 0.0), 1'000'000'000);
  const RigidTransform result =
      filter.update(pose(1.0, CV_PI / 2.0), 2'000'000'000);

  EXPECT_NEAR(result.translation.at<double>(0), 1.0, 1e-12);
  EXPECT_NEAR(rotationAngle(result), CV_PI / 2.0, 1e-12);
}

TEST(PoseFilterTest, UsesTimeAwareTranslationAndRotationSmoothing) {
  ExponentialPoseFilter filter(1.0);
  (void)filter.update(pose(0.0, 0.0), 1'000'000'000);
  const RigidTransform result =
      filter.update(pose(1.0, CV_PI / 2.0), 2'000'000'000);
  const double alpha = 1.0 - std::exp(-1.0);

  EXPECT_NEAR(result.translation.at<double>(0), alpha, 1e-12);
  EXPECT_NEAR(rotationAngle(result), alpha * CV_PI / 2.0, 1e-10);
  EXPECT_TRUE(filter.initialized());
}

TEST(PoseFilterTest, IsIndependentOfUpdateRateForAConstantMeasurement) {
  ExponentialPoseFilter oneStep(1.0);
  ExponentialPoseFilter twoSteps(1.0);
  (void)oneStep.update(pose(0.0, 0.0), 1'000'000'000);
  (void)twoSteps.update(pose(0.0, 0.0), 1'000'000'000);

  const RigidTransform once = oneStep.update(pose(1.0, 0.0), 2'000'000'000);
  (void)twoSteps.update(pose(1.0, 0.0), 1'500'000'000);
  const RigidTransform twice = twoSteps.update(pose(1.0, 0.0), 2'000'000'000);

  EXPECT_NEAR(once.translation.at<double>(0), twice.translation.at<double>(0),
              1e-12);
}

TEST(PoseFilterTest, InterpolatesRotationAcrossTheShortestArc) {
  ExponentialPoseFilter filter(1.0);
  (void)filter.update(pose(0.0, 170.0 * CV_PI / 180.0), 1'000'000'000);
  const auto halfLifeNanoseconds =
      static_cast<std::int64_t>(std::llround(std::log(2.0) * 1'000'000'000.0));
  const RigidTransform result = filter.update(
      pose(0.0, -170.0 * CV_PI / 180.0), 1'000'000'000 + halfLifeNanoseconds);

  EXPECT_NEAR(rotationAngle(result), CV_PI, 1e-8);
}

TEST(PoseFilterTest, HandlesEquivalentQuaternionSignAtBranchBoundary) {
  ExponentialPoseFilter filter(1.0);
  (void)filter.update(dominantAxisBoundaryPose(1.0 + 1.0e-12), 1'000'000'000);
  const RigidTransform result =
      filter.update(dominantAxisBoundaryPose(1.0 - 1.0e-12), 1'500'000'000);

  EXPECT_TRUE(result.isFinite());
  EXPECT_LT(cv::norm(result.rotation - dominantAxisBoundaryPose(1.0).rotation,
                     cv::NORM_INF),
            1.0e-8);
}

TEST(PoseFilterTest, ResetAndNonMonotonicTimeRestartFromMeasurement) {
  ExponentialPoseFilter filter(1.0);
  (void)filter.update(pose(0.0, 0.0), 2'000'000'000);
  const RigidTransform backwards = filter.update(pose(0.8, 0.4), 1'000'000'000);
  EXPECT_NEAR(backwards.translation.at<double>(0), 0.8, 1e-12);
  EXPECT_NEAR(rotationAngle(backwards), 0.4, 1e-10);

  filter.reset();
  EXPECT_FALSE(filter.initialized());
  const RigidTransform afterReset =
      filter.update(pose(0.6, 0.2), 3'000'000'000);
  EXPECT_NEAR(afterReset.translation.at<double>(0), 0.6, 1e-12);
  EXPECT_NEAR(rotationAngle(afterReset), 0.2, 1e-10);
}

TEST(PoseFilterTest, ChangingTimeConstantRestartsFromNextMeasurement) {
  ExponentialPoseFilter filter(0.25);
  (void)filter.update(pose(0.0, 0.0), 1'000'000'000);
  filter.setTimeConstantSeconds(0.75);
  EXPECT_FALSE(filter.initialized());

  const RigidTransform result = filter.update(pose(0.7, 0.3), 2'000'000'000);
  EXPECT_NEAR(result.translation.at<double>(0), 0.7, 1e-12);
  EXPECT_NEAR(rotationAngle(result), 0.3, 1e-10);
}

} // namespace
} // namespace hand_eye::core
