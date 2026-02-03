#pragma once

#include "geometry_msgs/msg/pose_stamped.hpp"
#include <ros_babel_fish/babel_fish.hpp>

using geometry_msgs::msg::PoseStamped;

using namespace ros_babel_fish;

inline void fillMessage(CompoundMessage &msg, const std::string &type,
                        const PoseStamped &pose) {
  if (type == "geometry_msgs/msg/PoseStamped") {
    msg["header"]["frame_id"] = pose.header.frame_id;
    msg["header"]["stamp"] = pose.header.stamp;
    msg["pose"]["position"]["x"] = pose.pose.position.x;
    msg["pose"]["position"]["y"] = pose.pose.position.y;
    msg["pose"]["position"]["z"] = pose.pose.position.z;

    msg["pose"]["orientation"]["x"] = pose.pose.orientation.x;
    msg["pose"]["orientation"]["y"] = pose.pose.orientation.y;
    msg["pose"]["orientation"]["z"] = pose.pose.orientation.z;
    msg["pose"]["orientation"]["w"] = pose.pose.orientation.w;
  } else if (type == "geometry_msgs/msg/Pose") {
    msg["header"]["frame_id"] = pose.header.frame_id;
    msg["header"]["stamp"] = pose.header.stamp;
    msg["position"]["x"] = pose.pose.position.x;
    msg["position"]["y"] = pose.pose.position.y;
    msg["position"]["z"] = pose.pose.position.z;
    msg["orientation"]["x"] = pose.pose.orientation.x;
    msg["orientation"]["y"] = pose.pose.orientation.y;
    msg["orientation"]["z"] = pose.pose.orientation.z;
    msg["orientation"]["w"] = pose.pose.orientation.w;
  }
}