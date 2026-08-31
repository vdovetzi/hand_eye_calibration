#include "lib/ros/target_marker.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hand_eye::ros {
namespace {

using Marker = visualization_msgs::msg::Marker;

Marker marker(const std::string &frame,
              const builtin_interfaces::msg::Time &stamp, const int id,
              const int type) {
  Marker result;
  result.header.frame_id = frame;
  result.header.stamp = stamp;
  result.ns = "hand_eye_aruco";
  result.id = id;
  result.type = type;
  result.action = Marker::ADD;
  result.pose.orientation.w = 1.0;
  result.lifetime.sec = 0;
  result.lifetime.nanosec = 750'000'000U;
  result.frame_locked = true;
  return result;
}

geometry_msgs::msg::Point point(const double x, const double y,
                                const double z) {
  geometry_msgs::msg::Point value;
  value.x = x;
  value.y = y;
  value.z = z;
  return value;
}

} // namespace

visualization_msgs::msg::MarkerArray
makeArucoMarkerArray(const core::PatternConfig &config,
                     const std::string &targetFrame,
                     const builtin_interfaces::msg::Time &stamp) {
  if (targetFrame.empty()) {
    throw std::invalid_argument("ArUco visualization frame must not be empty");
  }
  const cv::Mat modules = core::canonicalArucoMarker(config);
  const double size = config.markerSizeMeters;
  const double half = size / 2.0;
  const double cell = size / static_cast<double>(modules.cols);
  const double thickness = std::max(0.0008, size * 0.02);

  visualization_msgs::msg::MarkerArray array;

  Marker backing = marker(targetFrame, stamp, 0, Marker::CUBE);
  backing.pose.position.z = -thickness / 2.0;
  backing.scale.x = size;
  backing.scale.y = size;
  backing.scale.z = thickness;
  backing.color.r = 1.0F;
  backing.color.g = 1.0F;
  backing.color.b = 1.0F;
  backing.color.a = 1.0F;
  array.markers.push_back(std::move(backing));

  Marker black = marker(targetFrame, stamp, 1, Marker::CUBE_LIST);
  black.scale.x = cell;
  black.scale.y = cell;
  black.scale.z = thickness;
  black.color.r = 0.01F;
  black.color.g = 0.01F;
  black.color.b = 0.01F;
  black.color.a = 1.0F;
  for (int row = 0; row < modules.rows; ++row) {
    for (int column = 0; column < modules.cols; ++column) {
      if (modules.at<std::uint8_t>(row, column) != 0U) {
        continue;
      }
      black.points.push_back(point(
          -half + (static_cast<double>(column) + 0.5) * cell,
          half - (static_cast<double>(row) + 0.5) * cell, thickness / 2.0));
    }
  }
  array.markers.push_back(std::move(black));

  Marker label = marker(targetFrame, stamp, 2, Marker::TEXT_VIEW_FACING);
  label.pose.position.y = -half - size * 0.28;
  label.pose.position.z = size * 0.04;
  label.scale.z = std::max(0.012, size * 0.18);
  label.color.r = 0.1F;
  label.color.g = 0.8F;
  label.color.b = 1.0F;
  label.color.a = 1.0F;
  std::ostringstream text;
  text << "ArUco " << config.dictionarySize << "x" << config.dictionarySize
       << "/" << core::arucoDictionaryCapacity(config)
       << "  id=" << config.markerId << "  " << std::lround(size * 1000.0)
       << " mm";
  label.text = text.str();
  array.markers.push_back(std::move(label));

  Marker normal = marker(targetFrame, stamp, 3, Marker::ARROW);
  normal.points = {point(0.0, 0.0, 0.0), point(0.0, 0.0, size * 0.55)};
  normal.scale.x = size * 0.035;
  normal.scale.y = size * 0.075;
  normal.scale.z = size * 0.12;
  normal.color.r = 0.15F;
  normal.color.g = 0.35F;
  normal.color.b = 1.0F;
  normal.color.a = 1.0F;
  array.markers.push_back(std::move(normal));

  return array;
}

visualization_msgs::msg::MarkerArray
deleteTargetMarkers(const std::string &targetFrame,
                    const builtin_interfaces::msg::Time &stamp) {
  visualization_msgs::msg::MarkerArray array;
  Marker clear;
  clear.header.frame_id = targetFrame;
  clear.header.stamp = stamp;
  clear.ns = "hand_eye_aruco";
  clear.action = Marker::DELETEALL;
  clear.pose.orientation.w = 1.0;
  array.markers.push_back(std::move(clear));
  return array;
}

} // namespace hand_eye::ros
