#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace cv_ui {

constexpr const char *TEST_WINDOWNAME = "Test window";
constexpr const int32_t BASE_RADIUS = 6; // Base circle radius
constexpr const int32_t MAX_RADIUS = 10; // Max circle radius
constexpr const int32_t THICKNESS = -1;  // -1 if filled circle
constexpr const double BASE_TRANSPARENCY =
    0.3; // Base circles transparency (70% of transparency)
constexpr const double MAX_TRANSPARENCY = 0.8; // 20% of transparency
constexpr const double ACTIVATION_DISTANCE =
    30.0; // Distance between mouse and point to trigger
constexpr const double ANIMATION_SPEED = 1.0; // Speed of animation
const cv::Scalar CIRCLE_COLOR(0, 160, 255);   // Color for circles

class cvUI {
public:
  cvUI() = delete;
  cvUI(const cvUI &) = delete;
  cvUI operator=(const cvUI &) = delete;
  cvUI(cvUI &&) = delete;
  cvUI &&operator=(cvUI &&) = delete;

  cvUI(const std::vector<cv::Point2f> &pts, std::optional<size_t> &clicked,
       std::mutex &pt_grd, cv::Mat &img, std::mutex &img_grd)
      : points_(pts), image_(img), image_guard_(img_grd), point_guard_(pt_grd),
        clicked_index_(clicked) {
    cv::namedWindow(TEST_WINDOWNAME, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(TEST_WINDOWNAME, &cvUI::mouseCallback, this);

    initializeAnimation();
  }

  ~cvUI() { cv::destroyWindow(TEST_WINDOWNAME); }

  void show() {
    std::unique_lock<std::mutex> lock(image_guard_);
    if (image_.empty()) {
      return;
    }

    auto displayImage = image_.clone();

    lock.unlock();

    updateAnimation();

    drawPoints(displayImage);

    cv::imshow(TEST_WINDOWNAME, displayImage);

    // TODO: add opportunity to close app via 'q' or 'Q'
    [[maybe_unused]] int32_t key = cv::waitKey(1) & 0xFF;
  }

private:
  const std::vector<cv::Point2f> &points_;
  cv::Mat &image_;
  std::mutex &image_guard_;
  std::mutex &point_guard_;

  cv::Point currentMousePos_{-1, -1};
  std::optional<size_t> selectedPointIndex_;
  std::optional<size_t> &clicked_index_;

  std::vector<double> pointAlpha_;
  std::vector<double> pointRadius_;
  std::vector<double> targetAlpha_;
  std::vector<double> targetRadius_;

  static void mouseCallback(int event, int x, int y, int flags,
                            void *userdata) {
    cvUI *self = reinterpret_cast<cvUI *>(userdata);
    if (self) {
      self->handleMouseEvent(event, x, y, flags);
    }
  }

  void handleMouseEvent(int event, int x, int y, [[maybe_unused]] int flags) {
    currentMousePos_ = cv::Point(x, y);

    if (event == cv::EVENT_LBUTTONDOWN) {
      checkPointClick(x, y);
    }

    /*if (event == cv::EVENT_RBUTTONDOWN) {}*/

    /*if (event == cv::EVENT_MOUSEMOVE) {}*/
  }

  void checkPointClick(int x, int y) {
    if (points_.empty()) {
      return;
    }

    cv::Point clickPos(x, y);
    double minDistance = std::numeric_limits<double>::max();
    size_t closestIndex = 0;

    for (size_t i = 0; i < points_.size(); ++i) {
      double distance = cv::norm(points_[i] - cv::Point2f(clickPos));

      if (distance < minDistance) {
        minDistance = distance;
        closestIndex = i;
      }
    }

    if (minDistance <= MAX_RADIUS) {
      std::unique_lock<std::mutex> lock(point_guard_);
      clicked_index_ = closestIndex;
      lock.unlock();
    } else {
      std::cout << "Clicked outside of all points" << std::endl;
    }
  }

  void initializeAnimation() {
    if (points_.empty()) {
      return;
    }
    size_t numPoints = points_.size();

    if (numPoints != pointAlpha_.size()) {
      pointAlpha_.resize(numPoints, BASE_TRANSPARENCY);
      pointRadius_.resize(numPoints, BASE_RADIUS);
      targetAlpha_.resize(numPoints, BASE_TRANSPARENCY);
      targetRadius_.resize(numPoints, BASE_RADIUS);
    }
  }

  void updateAnimation() {
    if (points_.empty()) {
      return;
    }

    initializeAnimation();

    size_t numPoints = points_.size();

    for (size_t i = 0; i < numPoints; ++i) {
      double distance = cv::norm(points_[i] - cv::Point2f(currentMousePos_));

      if (distance < ACTIVATION_DISTANCE) {
        double factor = 1.0 - (distance / ACTIVATION_DISTANCE);
        factor = factor * factor;
        targetAlpha_[i] =
            BASE_TRANSPARENCY + (MAX_TRANSPARENCY - BASE_TRANSPARENCY) * factor;
        targetRadius_[i] = BASE_RADIUS + (MAX_RADIUS - BASE_RADIUS) * factor;
      } else {
        targetAlpha_[i] = BASE_TRANSPARENCY;
        targetRadius_[i] = BASE_RADIUS;
      }

      pointAlpha_[i] += (targetAlpha_[i] - pointAlpha_[i]) * ANIMATION_SPEED;
      pointRadius_[i] += (targetRadius_[i] - pointRadius_[i]) * ANIMATION_SPEED;

      pointAlpha_[i] =
          std::clamp(pointAlpha_[i], BASE_TRANSPARENCY, MAX_TRANSPARENCY);
      pointRadius_[i] =
          std::clamp(pointRadius_[i], static_cast<double>(BASE_RADIUS),
                     static_cast<double>(MAX_RADIUS));
    }
  }

  void drawPoints(cv::Mat &image) {
    if (points_.empty()) {
      return;
    }

    size_t numPoints = points_.size();

    cv::Mat overlay = cv::Mat::zeros(image.size(), image.type());

    for (size_t i = 0; i < numPoints; ++i) {
      const cv::Point2f &point = points_[i];
      double alpha = pointAlpha_[i];
      int radius = static_cast<int>(std::round(pointRadius_[i]));

      cv::Scalar color(CIRCLE_COLOR[0], CIRCLE_COLOR[1],
                       CIRCLE_COLOR[2] * alpha);

      cv::circle(overlay, point, radius, color, THICKNESS);
    }

    for (int y = 0; y < image.rows; ++y) {
      for (int x = 0; x < image.cols; ++x) {
        cv::Vec3b overlayVal = overlay.at<cv::Vec3b>(y, x);
        if (overlayVal[0] > 0 || overlayVal[1] > 0 || overlayVal[2] > 0) {
          cv::Vec3b &pixel = image.at<cv::Vec3b>(y, x);
          float alpha = overlayVal[2] / 255.0f;
          pixel[0] =
              cv::saturate_cast<uchar>(pixel[0] * (1 - alpha) + overlayVal[0]);
          pixel[1] =
              cv::saturate_cast<uchar>(pixel[1] * (1 - alpha) + overlayVal[1]);
          pixel[2] =
              cv::saturate_cast<uchar>(pixel[2] * (1 - alpha) + overlayVal[2]);
        }
      }
    }
  }
};
}; // namespace cv_ui
