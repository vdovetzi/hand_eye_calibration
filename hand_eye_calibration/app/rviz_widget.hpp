#pragma once

#include <QWidget>

#include <functional>
#include <memory>

class QString;

namespace hand_eye::app {

class RvizWidget final : public QWidget {
public:
  explicit RvizWidget(QWidget *parent = nullptr);
  ~RvizWidget() override;

  RvizWidget(const RvizWidget &) = delete;
  RvizWidget &operator=(const RvizWidget &) = delete;

  void configureFrames(const QString &fixedFrame,
                       const QString &robotDescriptionTopic,
                       const QString &cameraFrame, const QString &targetFrame,
                       const QString &gripperFrame,
                       const QString &targetMarkersTopic,
                       const QString &cameraTransformSource = {},
                       bool cameraTransformActive = false,
                       bool gripperTransformActive = false,
                       bool targetTransformActive = false);
  void initializeVisualization();
  void setActive(bool active);
  void setExpandHandler(std::function<void(bool)> handler);
  void setExpanded(bool expanded);
  void queueRender();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace hand_eye::app
