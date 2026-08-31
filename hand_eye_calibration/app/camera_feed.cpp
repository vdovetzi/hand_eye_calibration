#include "app/camera_feed.hpp"

#include "lib/core/target.hpp"

#include <QImage>
#include <QLabel>
#include <QPixmap>
#include <QTimer>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace hand_eye::app {
namespace {

constexpr const char *kImageType = "sensor_msgs/msg/Image";
constexpr const char *kCompressedImageType = "sensor_msgs/msg/CompressedImage";

std::optional<std::string> topicType(const rclcpp::Node::SharedPtr &node,
                                     const std::string &topicName) {
  const auto topics = node->get_topic_names_and_types();
  const auto found = topics.find(topicName);
  if (found == topics.end() || found->second.empty()) {
    return std::nullopt;
  }
  return found->second.front();
}

struct FeedState {
  std::atomic_bool stop{false};
  std::mutex throttleMutex;
  std::chrono::steady_clock::time_point nextFrameTime{};
  std::condition_variable frameReady;
  std::mutex frameMutex;
  cv::Mat latestFrame;
  bool hasFrame = false;
  std::mutex renderMutex;
  QImage renderedImage;
  std::unique_ptr<core::TargetDetector> detector;
};

bool shouldAcceptFrame(const std::shared_ptr<FeedState> &state) {
  std::lock_guard lock(state->throttleMutex);
  const auto now = std::chrono::steady_clock::now();
  if (now < state->nextFrameTime) {
    return false;
  }
  state->nextFrameTime = now + std::chrono::milliseconds(33);
  return true;
}

void pushFrame(const std::shared_ptr<FeedState> &state, const cv::Mat &image) {
  if (image.empty() || state->stop.load()) {
    return;
  }
  {
    std::lock_guard lock(state->frameMutex);
    state->latestFrame = image.clone();
    state->hasFrame = true;
  }
  state->frameReady.notify_one();
}

void processFrames(const std::shared_ptr<FeedState> &state) {
  while (!state->stop.load()) {
    cv::Mat image;
    {
      std::unique_lock lock(state->frameMutex);
      state->frameReady.wait(
          lock, [&state]() { return state->stop.load() || state->hasFrame; });
      if (state->stop.load()) {
        return;
      }
      image = state->latestFrame.clone();
      state->hasFrame = false;
    }

    if (state->detector) {
      const auto observation = state->detector->detect(image);
      if (observation) {
        state->detector->draw(image, *observation);
      }
    }

    cv::Mat rgb;
    if (image.channels() == 1) {
      cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    } else {
      cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    }
    const QImage view(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                      QImage::Format_RGB888);
    std::lock_guard lock(state->renderMutex);
    state->renderedImage = view.copy();
  }
}

} // namespace

class CameraFeed::Impl final {
public:
  Impl(rclcpp::Node::SharedPtr node, std::string topic, std::string type,
       QLabel *label, const std::string &patternInfo)
      : state_(std::make_shared<FeedState>()) {
    if (type.empty()) {
      type = topicType(node, topic).value_or(kImageType);
    }
    if (!patternInfo.empty()) {
      try {
        state_->detector = std::make_unique<core::TargetDetector>(
            core::parsePattern(patternInfo));
      } catch (const std::exception &) {
        state_->detector.reset();
      }
    }

    worker_ = std::thread([state = state_]() { processFrames(state); });
    timer_ = new QTimer(label);
    timer_->setInterval(33);
    std::weak_ptr<FeedState> weakState = state_;
    QObject::connect(timer_, &QTimer::timeout, label, [weakState, label]() {
      const auto state = weakState.lock();
      if (!state) {
        return;
      }
      QImage image;
      {
        std::lock_guard lock(state->renderMutex);
        if (state->renderedImage.isNull()) {
          return;
        }
        image = state->renderedImage;
      }
      label->setPixmap(QPixmap::fromImage(image).scaled(
          label->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
    });
    timer_->start();

    if (type == kCompressedImageType) {
      const std::string compressedTopic =
          topic.ends_with("/compressed") ? topic : topic + "/compressed";
      compressedSubscription_ =
          node->create_subscription<sensor_msgs::msg::CompressedImage>(
              compressedTopic, rclcpp::SensorDataQoS(),
              [weakState](
                  const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
                const auto state = weakState.lock();
                if (!state || !shouldAcceptFrame(state)) {
                  return;
                }
                const cv::Mat encoded(
                    1, static_cast<int>(msg->data.size()), CV_8UC1,
                    const_cast<std::uint8_t *>(msg->data.data()));
                pushFrame(state, cv::imdecode(encoded, cv::IMREAD_COLOR));
              });
    } else {
      imageSubscription_ = node->create_subscription<sensor_msgs::msg::Image>(
          topic, rclcpp::SensorDataQoS(),
          [weakState](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
            const auto state = weakState.lock();
            if (!state || !shouldAcceptFrame(state)) {
              return;
            }
            try {
              pushFrame(state, cv_bridge::toCvCopy(
                                   *msg, sensor_msgs::image_encodings::BGR8)
                                   ->image);
            } catch (const std::exception &) {
            }
          });
    }
  }

  ~Impl() {
    imageSubscription_.reset();
    compressedSubscription_.reset();
    if (timer_) {
      timer_->stop();
      timer_->disconnect();
      delete timer_;
      timer_ = nullptr;
    }
    state_->stop.store(true);
    state_->frameReady.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  std::shared_ptr<FeedState> state_;
  QTimer *timer_ = nullptr;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSubscription_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
      compressedSubscription_;
  std::thread worker_;
};

CameraFeed::CameraFeed(rclcpp::Node::SharedPtr node, std::string topic,
                       std::string type, QLabel *label,
                       const std::string &patternInfo)
    : impl_(std::make_unique<Impl>(std::move(node), std::move(topic),
                                   std::move(type), label, patternInfo)) {}

CameraFeed::~CameraFeed() = default;

} // namespace hand_eye::app
