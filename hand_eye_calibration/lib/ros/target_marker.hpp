#pragma once

#include "lib/core/target.hpp"

#include <builtin_interfaces/msg/time.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <string>

namespace hand_eye::ros {

inline constexpr char kTargetMarkersTopic[] =
    "/hand_eye_backend/target_markers";

[[nodiscard]] visualization_msgs::msg::MarkerArray
makeArucoMarkerArray(const core::PatternConfig &config,
                     const std::string &targetFrame,
                     const builtin_interfaces::msg::Time &stamp);

[[nodiscard]] visualization_msgs::msg::MarkerArray
deleteTargetMarkers(const std::string &targetFrame,
                    const builtin_interfaces::msg::Time &stamp);

} // namespace hand_eye::ros
