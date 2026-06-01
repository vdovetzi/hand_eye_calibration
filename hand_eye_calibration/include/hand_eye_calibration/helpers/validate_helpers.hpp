#pragma once

#include "geometry_msgs/msg/pose_stamped.hpp"
#include <ros_babel_fish/babel_fish.hpp>

using geometry_msgs::msg::PoseStamped;

namespace validate_helpers {

void fillMessage(ros_babel_fish::CompoundMessage &msg, const std::string &type,
                 const PoseStamped &pose);

} // namespace validate_helpers
