#include "lib/core/capture_policy.hpp"

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>

#include <cstdint>

namespace hand_eye::core {
namespace {

constexpr std::int64_t kMillisecond = 1'000'000;

RigidTransform pose(const double x = 0.0, const double angleRadians = 0.0) {
  cv::Mat rotation;
  cv::Rodrigues(cv::Mat(cv::Vec3d(0.0, 0.0, angleRadians)), rotation);
  return {rotation, cv::Mat(cv::Vec3d(x, 0.0, 0.0))};
}

CaptureRequest validRequest() {
  CaptureRequest request;
  request.image =
      ImageSampleMetadata{1005 * kMillisecond, "camera", 640, 480, true};
  for (std::int64_t time = 400; time <= 1000; time += 100) {
    request.robotHistory.push_back(
        {time * kMillisecond, pose(), "base", "tool", true});
  }
  request.cameraInfo = CameraInfoMetadata{"camera", 640, 480};
  return request;
}

CapturePolicy policy() {
  CapturePolicyOptions options;
  options.requireCameraInfo = true;
  options.robotBaseFrame = "base";
  options.robotEffectorFrame = "tool";
  options.cameraFrame = "camera";
  return CapturePolicy(options);
}

TEST(CapturePolicyTest, AcceptsNearestSynchronizedStationaryPose) {
  const CaptureDecision decision = policy().evaluate(validRequest());

  ASSERT_TRUE(decision.accepted) << decision.message;
  ASSERT_TRUE(decision.matchedRobotPose.has_value());
  EXPECT_EQ(decision.matchedRobotPose->timestampNanoseconds,
            1000 * kMillisecond);
  EXPECT_DOUBLE_EQ(decision.timestampDeltaMilliseconds, 5.0);
}

TEST(CapturePolicyTest, RequiresSourceTimestampsByDefault) {
  CaptureRequest request = validRequest();
  request.image->hasSourceTimestamp = false;
  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::UNSTAMPED_SAMPLE);

  request = validRequest();
  for (TimedRobotPose &robot : request.robotHistory) {
    robot.hasSourceTimestamp = false;
  }
  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::UNSTAMPED_SAMPLE);
}

TEST(CapturePolicyTest, RejectsTimestampAndFrameMismatches) {
  CaptureRequest request = validRequest();
  request.image->timestampNanoseconds = 1100 * kMillisecond;
  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::TIMESTAMP_MISMATCH);

  request = validRequest();
  request.robotHistory.back().parentFrame = "map";
  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::FRAME_MISMATCH);
}

TEST(CapturePolicyTest, ValidatesCameraInfoPresenceAndResolution) {
  CaptureRequest request = validRequest();
  request.cameraInfo.reset();
  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::CAMERA_INFO_MISSING);

  request = validRequest();
  request.cameraInfo->calibrated = false;
  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::CAMERA_INFO_INVALID);

  request = validRequest();
  request.cameraInfo->width = 1280;
  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::CAMERA_INFO_SIZE_MISMATCH);
}

TEST(CapturePolicyTest, RejectsMotionDuringSettlingWindow) {
  CaptureRequest request = validRequest();
  request.robotHistory[4].pose = pose(0.01);

  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::ROBOT_NOT_SETTLED);
}

TEST(CapturePolicyTest, RequiresEnoughHistoryForSettlingWindow) {
  CaptureRequest request = validRequest();
  request.robotHistory.erase(request.robotHistory.begin(),
                             request.robotHistory.begin() + 2);

  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::ROBOT_NOT_SETTLED);
}

TEST(CapturePolicyTest, RejectsNearDuplicatePose) {
  CaptureRequest request = validRequest();
  request.acceptedRobotPoses.push_back(pose(0.001, 0.01));

  EXPECT_EQ(policy().evaluate(request).reason,
            CaptureRejectionReason::DUPLICATE_POSE);
}

TEST(CapturePolicyTest, CompatibilityModeAllowsUnstampedSamples) {
  CaptureRequest request = validRequest();
  request.image->hasSourceTimestamp = false;
  for (TimedRobotPose &robot : request.robotHistory) {
    robot.hasSourceTimestamp = false;
  }
  CapturePolicyOptions options = policy().options();
  options.requireStampedSamples = false;

  EXPECT_TRUE(CapturePolicy(options).evaluate(request).accepted);
}

TEST(CapturePolicyTest, CompatibilityModeDoesNotMixReceiptAndSourceClocks) {
  CaptureRequest request = validRequest();
  for (std::size_t index = 0; index < request.robotHistory.size(); ++index) {
    request.robotHistory[index].hasSourceTimestamp = false;
    request.robotHistory[index].timestampNanoseconds =
        (index == 0 ? 1005 : 2100 + static_cast<std::int64_t>(index) * 100) *
        kMillisecond;
  }
  CapturePolicyOptions options = policy().options();
  options.requireStampedSamples = false;

  const CaptureDecision decision = CapturePolicy(options).evaluate(request);

  ASSERT_TRUE(decision.accepted) << decision.message;
  ASSERT_TRUE(decision.matchedRobotPose);
  EXPECT_EQ(decision.matchedRobotPose->timestampNanoseconds,
            request.robotHistory.back().timestampNanoseconds);
  EXPECT_DOUBLE_EQ(decision.timestampDeltaMilliseconds, 0.0);
}

} // namespace
} // namespace hand_eye::core
