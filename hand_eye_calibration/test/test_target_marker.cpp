#include "lib/ros/target_marker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hand_eye::ros {
namespace {

TEST(TargetMarkerTest, ReconstructsCanonicalMarkerWithoutMirroring) {
  const core::PatternConfig config = core::parsePattern("1 4 50 0.081");
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 12;
  stamp.nanosec = 345U;

  const auto array = makeArucoMarkerArray(config, "aruco_50", stamp);
  ASSERT_EQ(array.markers.size(), 4U);
  const auto &backing = array.markers[0];
  const auto &black = array.markers[1];
  const auto &label = array.markers[2];
  const auto &normal = array.markers[3];
  EXPECT_EQ(backing.type, visualization_msgs::msg::Marker::CUBE);
  EXPECT_DOUBLE_EQ(backing.scale.x, config.markerSizeMeters);
  EXPECT_DOUBLE_EQ(backing.scale.y, config.markerSizeMeters);
  EXPECT_LT(backing.pose.position.z, 0.0);
  EXPECT_EQ(black.type, visualization_msgs::msg::Marker::CUBE_LIST);
  EXPECT_GT(black.scale.z, 0.0);
  EXPECT_TRUE(black.frame_locked);
  EXPECT_EQ(black.header.frame_id, "aruco_50");
  EXPECT_EQ(black.header.stamp, stamp);

  const cv::Mat modules = core::canonicalArucoMarker(config);
  cv::Mat reconstructed(modules.size(), CV_8UC1, cv::Scalar(255));
  const double half = config.markerSizeMeters / 2.0;
  const double cell = config.markerSizeMeters / modules.cols;
  for (const auto &position : black.points) {
    const int column =
        static_cast<int>(std::lround((position.x + half) / cell - 0.5));
    const int row =
        static_cast<int>(std::lround((half - position.y) / cell - 0.5));
    ASSERT_GE(row, 0);
    ASSERT_LT(row, modules.rows);
    ASSERT_GE(column, 0);
    ASSERT_LT(column, modules.cols);
    EXPECT_GT(position.z, 0.0);
    reconstructed.at<std::uint8_t>(row, column) = 0U;
  }
  EXPECT_EQ(cv::countNonZero(reconstructed != modules), 0);
  EXPECT_NE(label.text.find("id=50"), std::string::npos);
  EXPECT_NE(label.text.find("4x4/100"), std::string::npos);
  ASSERT_EQ(normal.points.size(), 2U);
  EXPECT_GT(normal.points[1].z, 0.0);
}

TEST(TargetMarkerTest, RejectsMissingFrameAndBuildsDeleteAll) {
  const core::PatternConfig config = core::parsePattern("1 4 7 0.04");
  EXPECT_THROW(makeArucoMarkerArray(config, "", {}), std::invalid_argument);

  const auto clear = deleteTargetMarkers("aruco_7", {});
  ASSERT_EQ(clear.markers.size(), 1U);
  EXPECT_EQ(clear.markers.front().action,
            visualization_msgs::msg::Marker::DELETEALL);
}

} // namespace
} // namespace hand_eye::ros
