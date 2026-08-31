#include "app/rviz_widget.hpp"

#include <QButtonGroup>
#include <QCoreApplication>
#include <QEventLoop>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSizePolicy>
#include <QString>
#include <QToolButton>
#include <QVBoxLayout>

#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rviz_common/display.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/properties/property.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/tool.hpp>
#include <rviz_common/tool_manager.hpp>
#include <rviz_common/view_controller.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_common/visualization_manager.hpp>
#include <rviz_rendering/render_window.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace hand_eye::app {
namespace {

void configureEmbeddedRvizLogging() {
  static std::once_flag loggingHandlersInstalled;
  std::call_once(loggingHandlersInstalled, []() {
    const auto discard = [](const std::string &, const std::string &,
                            std::size_t) {};
    const auto logger = rclcpp::get_logger("hand_eye_frontend.embedded_rviz");
    const auto warning = [logger](const std::string &message,
                                  const std::string &, std::size_t) {
      RCLCPP_WARN(logger, "Embedded RViz: %s", message.c_str());
    };
    const auto error = [logger](const std::string &message, const std::string &,
                                std::size_t) {
      RCLCPP_ERROR(logger, "Embedded RViz: %s", message.c_str());
    };

    rviz_common::set_logging_handlers(discard, discard, warning, error);
    rviz_common::install_rviz_rendering_log_handlers();
  });
}

class RvizNode final
    : public rviz_common::ros_integration::RosNodeAbstractionIface {
public:
  RvizNode()
      : node_(std::make_shared<rclcpp::Node>(
            "hand_eye_embedded_rviz",
            rclcpp::NodeOptions().use_global_arguments(false))) {}

  std::string get_node_name() const override { return node_->get_name(); }

  std::map<std::string, std::vector<std::string>>
  get_topic_names_and_types() const override {
    return node_->get_topic_names_and_types();
  }

  rclcpp::Node::SharedPtr get_raw_node() override { return node_; }

private:
  rclcpp::Node::SharedPtr node_;
};

void setDisplayProperty(rviz_common::Display *display, const QString &name,
                        const QVariant &value) {
  if (!display) {
    return;
  }
  if (auto *property = display->subProp(name)) {
    property->setValue(value);
  }
}

void setDisplayChildProperty(rviz_common::Display *display,
                             const QString &parentName,
                             const QString &childName, const QVariant &value) {
  if (!display) {
    return;
  }
  auto *parent = display->subProp(parentName);
  if (parent) {
    if (auto *child = parent->subProp(childName)) {
      child->setValue(value);
    }
  }
}

void setRvizProperty(rviz_common::properties::Property *parent,
                     const QString &name, const QVariant &value) {
  if (!parent) {
    return;
  }
  if (auto *property = parent->subProp(name)) {
    property->setValue(value);
  }
}

void setNestedProperty(rviz_common::properties::Property *parent,
                       const QString &groupName, const QString &name,
                       const QVariant &value) {
  if (!parent) {
    return;
  }
  if (auto *group = parent->subProp(groupName)) {
    setRvizProperty(group, name, value);
  }
}

QString usableFrame(const QString &requested, const QString &fallback) {
  const QString frame = requested.trimmed();
  return frame.isEmpty() ? fallback : frame;
}

} // namespace

class RvizWidget::Impl final {
public:
  explicit Impl(RvizWidget &widget) : widget_(widget) {
    layout_ = new QVBoxLayout(&widget_);
    layout_->setContentsMargins(0, 0, 0, 0);
    layout_->setSpacing(4);
    layout_->setSizeConstraint(QLayout::SetNoConstraint);
    widget_.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    auto *header = new QWidget(&widget_);
    header->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    auto *headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(8, 2, 8, 2);
    status_ = new QLabel("Initializing embedded RViz...", header);
    status_->setStyleSheet("color: #5b6470;");
    status_->setMinimumWidth(120);
    status_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    headerLayout->addWidget(status_, 1);

    toolsBar_ = new QWidget(header);
    toolsLayout_ = new QHBoxLayout(toolsBar_);
    toolsLayout_->setContentsMargins(0, 0, 0, 0);
    toolsLayout_->setSpacing(2);
    toolsPlaceholder_ = new QLabel("3D tools loading...", toolsBar_);
    toolsPlaceholder_->setStyleSheet("color: #7b8490;");
    toolsLayout_->addWidget(toolsPlaceholder_);
    toolButtons_ = new QButtonGroup(toolsBar_);
    toolButtons_->setExclusive(true);
    headerLayout->addWidget(toolsBar_);

    auto *reset = new QPushButton("Reset view", header);
    headerLayout->addWidget(reset);
    expand_ = new QPushButton("Expand (F11)", header);
    expand_->setToolTip("Use the full window for interactive RViz (F11)");
    headerLayout->addWidget(expand_);
    layout_->addWidget(header);
    QObject::connect(reset, &QPushButton::clicked, &widget_,
                     [this]() { resetView(); });
    QObject::connect(expand_, &QPushButton::clicked, &widget_, [this]() {
      const bool requested = !expanded_;
      if (expandHandler_) {
        expandHandler_(requested);
      } else {
        setExpanded(requested);
      }
    });
  }

  void initialize() {
    if (initializationStarted_ || !widget_.isVisible()) {
      return;
    }
    configureEmbeddedRvizLogging();
    initializationStarted_ = true;
    try {
      rvizNode_ = std::make_shared<RvizNode>();
      renderPanel_ = new rviz_common::RenderPanel(&widget_);
      renderPanel_->setMinimumSize(320, 240);
      renderPanel_->setSizePolicy(QSizePolicy::Expanding,
                                  QSizePolicy::Expanding);
      renderPanel_->setFocusPolicy(Qt::StrongFocus);
      layout_->addWidget(renderPanel_, 1);
      renderPanel_->show();
      layout_->activate();
      QCoreApplication::processEvents(QEventLoop::AllEvents);
      renderPanel_->getRenderWindow()->initialize();
      QCoreApplication::processEvents(QEventLoop::AllEvents);

      manager_ = std::make_unique<rviz_common::VisualizationManager>(
          renderPanel_, rvizNode_, nullptr,
          rvizNode_->get_raw_node()->get_clock());
      QCoreApplication::processEvents(QEventLoop::AllEvents);
      renderPanel_->initialize(manager_.get());
      QCoreApplication::processEvents(QEventLoop::AllEvents);
      manager_->initialize();
      QCoreApplication::processEvents(QEventLoop::AllEvents);

      initializeTools();

      grid_ =
          manager_->createDisplay("rviz_default_plugins/Grid", "Grid", true);
      setDisplayProperty(grid_, "Cell Size", 0.05);
      setDisplayProperty(grid_, "Plane Cell Count", 20);
      setDisplayProperty(grid_, "Alpha", 0.35);

      robot_ = manager_->createDisplay("rviz_default_plugins/RobotModel",
                                       "Sagittarius Robot", true);
      setDisplayProperty(robot_, "Description Source", "Topic");
      setDisplayProperty(robot_, "Visual Enabled", true);
      setDisplayProperty(robot_, "Collision Enabled", false);
      setDisplayProperty(robot_, "Alpha", 1.0);
      setDisplayChildProperty(robot_, "Description Topic", "Durability Policy",
                              "Transient Local");

      gripperAxes_ = manager_->createDisplay("rviz_default_plugins/Axes",
                                             "Gripper pose", true);
      setDisplayProperty(gripperAxes_, "Length", 0.16);
      setDisplayProperty(gripperAxes_, "Radius", 0.008);

      cameraAxes_ = manager_->createDisplay("rviz_default_plugins/Axes",
                                            "Camera frame", true);
      setDisplayProperty(cameraAxes_, "Length", 0.20);
      setDisplayProperty(cameraAxes_, "Radius", 0.009);

      targetAxes_ = manager_->createDisplay("rviz_default_plugins/Axes",
                                            "Calibration target", true);
      setDisplayProperty(targetAxes_, "Length", 0.12);
      setDisplayProperty(targetAxes_, "Radius", 0.006);

      targetMarker_ = manager_->createDisplay(
          "rviz_default_plugins/MarkerArray", "Detected ArUco marker", true);

      ready_ = true;
      applyDisplayConfiguration();
      updateStatus();
      if (active_) {
        startManagerUpdate();
      }
      resetView();
    } catch (const std::exception &error) {
      status_->setText("Embedded RViz unavailable: " +
                       QString::fromUtf8(error.what()));
      status_->setStyleSheet("color: #b42318;");
    }
  }

  ~Impl() {
    if (manager_) {
      stopManagerUpdate();
      manager_->removeAllDisplays();
      manager_.reset();
    }
    delete renderPanel_;
    renderPanel_ = nullptr;
    rvizNode_.reset();
  }

  void configureFrames(
      const QString &fixedFrame, const QString &robotDescriptionTopic,
      const QString &cameraFrame, const QString &targetFrame,
      const QString &gripperFrame, const QString &targetMarkersTopic,
      const QString &cameraTransformSource, const bool cameraTransformActive,
      const bool gripperTransformActive, const bool targetTransformActive) {
    if (fixedFrame_ == fixedFrame &&
        robotDescriptionTopic_ == robotDescriptionTopic &&
        cameraFrame_ == cameraFrame && targetFrame_ == targetFrame &&
        gripperFrame_ == gripperFrame &&
        targetMarkersTopic_ == targetMarkersTopic &&
        cameraTransformSource_ == cameraTransformSource &&
        cameraTransformActive_ == cameraTransformActive &&
        gripperTransformActive_ == gripperTransformActive &&
        targetTransformActive_ == targetTransformActive) {
      return;
    }

    const bool fixedFrameChanged = fixedFrame_ != fixedFrame;
    const bool robotDescriptionChanged =
        robotDescriptionTopic_ != robotDescriptionTopic;
    const bool cameraFrameChanged = cameraFrame_ != cameraFrame;
    const bool targetFrameChanged = targetFrame_ != targetFrame;
    const bool gripperFrameChanged = gripperFrame_ != gripperFrame;
    const bool targetMarkersTopicChanged =
        targetMarkersTopic_ != targetMarkersTopic;
    fixedFrame_ = fixedFrame;
    robotDescriptionTopic_ = robotDescriptionTopic;
    cameraFrame_ = cameraFrame;
    targetFrame_ = targetFrame;
    gripperFrame_ = gripperFrame;
    targetMarkersTopic_ = targetMarkersTopic;
    cameraTransformSource_ = cameraTransformSource;
    cameraTransformActive_ = cameraTransformActive;
    gripperTransformActive_ = gripperTransformActive;
    targetTransformActive_ = targetTransformActive;

    if (!ready_) {
      return;
    }
    if (fixedFrameChanged) {
      manager_->setFixedFrame(usableFrame(fixedFrame_, "sgr532/base_link"));
    }
    if (robotDescriptionChanged) {
      setDisplayProperty(
          robot_, "Description Topic",
          usableFrame(robotDescriptionTopic_, "/sgr532/robot_description"));
    }
    if (gripperFrameChanged) {
      setDisplayProperty(gripperAxes_, "Reference Frame",
                         usableFrame(gripperFrame_, "hand_eye_gripper_pose"));
    }
    if (cameraFrameChanged) {
      setDisplayProperty(cameraAxes_, "Reference Frame",
                         usableFrame(cameraFrame_, "realsense"));
    }
    if (targetFrameChanged) {
      setDisplayProperty(targetAxes_, "Reference Frame",
                         usableFrame(targetFrame_, "calibration_target"));
    }
    if (targetMarkersTopicChanged) {
      setDisplayProperty(
          targetMarker_, "Topic",
          usableFrame(targetMarkersTopic_, "/hand_eye_backend/target_markers"));
    }
    updateStatus();
    if (fixedFrameChanged || robotDescriptionChanged || cameraFrameChanged ||
        targetFrameChanged || gripperFrameChanged ||
        targetMarkersTopicChanged) {
      queueRender();
    }
  }

  void setActive(const bool active) {
    if (active_ == active) {
      return;
    }
    active_ = active;
    if (!ready_) {
      return;
    }
    if (active_) {
      startManagerUpdate();
    } else {
      stopManagerUpdate();
    }
  }

  void setExpandHandler(std::function<void(bool)> handler) {
    expandHandler_ = std::move(handler);
  }

  void setExpanded(const bool expanded) {
    expanded_ = expanded;
    expand_->setText(expanded ? "Restore (F11)" : "Expand (F11)");
    expand_->setToolTip(expanded
                            ? "Return to the calibration workflow (F11)"
                            : "Use the full window for interactive RViz (F11)");
  }

  void queueRender() {
    if (manager_ && updateRunning_) {
      manager_->queueRender();
    }
  }

private:
  void initializeTools() {
    if (!manager_ || !manager_->getToolManager()) {
      return;
    }
    auto *manager = manager_->getToolManager();
    manager->removeAll();

    struct ToolSpec {
      const char *classId;
      const char *fallbackLabel;
    };
    const std::vector<ToolSpec> specs = {
        {"rviz_default_plugins/MoveCamera", "Move camera"},
        {"rviz_default_plugins/Interact", "Interact"},
        {"rviz_default_plugins/Select", "Select"},
        {"rviz_default_plugins/FocusCamera", "Focus"},
    };

    rviz_common::Tool *moveCamera = nullptr;
    for (const auto &spec : specs) {
      auto *tool = manager->addTool(spec.classId);
      if (!tool) {
        continue;
      }
      if (!moveCamera) {
        moveCamera = tool;
      }
      auto *button = new QToolButton(toolsBar_);
      const QString toolName = tool->getName().trimmed();
      button->setText(toolName.isEmpty() ? spec.fallbackLabel : toolName);
      button->setIcon(tool->getIcon());
      button->setToolButtonStyle(Qt::ToolButtonIconOnly);
      button->setCheckable(true);
      button->setAutoRaise(true);
      button->setAccessibleName(button->text());
      button->setToolTip(button->text() + ": " + tool->getDescription());
      toolButtons_->addButton(button);
      toolsLayout_->addWidget(button);
      tools_.emplace_back(button, tool);
      QObject::connect(button, &QToolButton::clicked, &widget_,
                       [manager, tool]() { manager->setCurrentTool(tool); });
    }
    toolsPlaceholder_->setVisible(tools_.empty());

    QObject::connect(manager, &rviz_common::ToolManager::toolChanged, &widget_,
                     [this](rviz_common::Tool *active) {
                       for (const auto &[button, tool] : tools_) {
                         button->setChecked(tool == active);
                       }
                     });
    if (moveCamera) {
      manager->setDefaultTool(moveCamera);
      manager->setCurrentTool(moveCamera);
    }
  }

  void applyDisplayConfiguration() {
    if (!ready_) {
      return;
    }
    const QString fixed = usableFrame(fixedFrame_, "sgr532/base_link");
    manager_->setFixedFrame(fixed);
    setDisplayProperty(
        robot_, "Description Topic",
        usableFrame(robotDescriptionTopic_, "/sgr532/robot_description"));
    setDisplayProperty(gripperAxes_, "Reference Frame",
                       usableFrame(gripperFrame_, "hand_eye_gripper_pose"));
    setDisplayProperty(cameraAxes_, "Reference Frame",
                       usableFrame(cameraFrame_, "realsense"));
    setDisplayProperty(targetAxes_, "Reference Frame",
                       usableFrame(targetFrame_, "calibration_target"));
    setDisplayProperty(
        targetMarker_, "Topic",
        usableFrame(targetMarkersTopic_, "/hand_eye_backend/target_markers"));
  }

  void updateStatus() {
    const QString fixed = usableFrame(fixedFrame_, "sgr532/base_link");
    QString cameraState = cameraTransformSource_.trimmed();
    if (cameraState.isEmpty()) {
      cameraState = cameraTransformActive_ ? "live" : "waiting";
    } else if (!cameraTransformActive_) {
      cameraState += ", waiting";
    }
    const QString gripperState = gripperTransformActive_ ? "live" : "waiting";
    const QString status =
        QString("Fixed: %1 | camera: %2 (%3) | gripper: %4 (%5) "
                "| target: %6 (%7)")
            .arg(fixed, usableFrame(cameraFrame_, "realsense"), cameraState,
                 usableFrame(gripperFrame_, "hand_eye_gripper_pose"),
                 gripperState, usableFrame(targetFrame_, "calibration_target"),
                 targetTransformActive_ ? "live" : "waiting");
    if (status_->text() != status) {
      status_->setText(status);
      status_->setToolTip(status);
    }
  }

  void startManagerUpdate() {
    if (!manager_ || updateRunning_) {
      return;
    }
    manager_->startUpdate();
    updateRunning_ = true;
    manager_->queueRender();
  }

  void stopManagerUpdate() {
    if (!manager_ || !updateRunning_) {
      return;
    }
    manager_->stopUpdate();
    updateRunning_ = false;
  }

  void resetView() {
    if (!manager_ || !manager_->getViewManager() ||
        !manager_->getViewManager()->getCurrent()) {
      return;
    }
    rviz_common::ViewController *view =
        manager_->getViewManager()->getCurrent();
    setRvizProperty(view, "Distance", 1.15);
    setRvizProperty(view, "Pitch", 0.62);
    setRvizProperty(view, "Yaw", 0.78);
    setNestedProperty(view, "Focal Point", "X", 0.25);
    setNestedProperty(view, "Focal Point", "Y", 0.0);
    setNestedProperty(view, "Focal Point", "Z", 0.2);
    queueRender();
  }

  RvizWidget &widget_;
  QVBoxLayout *layout_{nullptr};
  QLabel *status_{nullptr};
  QWidget *toolsBar_{nullptr};
  QHBoxLayout *toolsLayout_{nullptr};
  QLabel *toolsPlaceholder_{nullptr};
  QButtonGroup *toolButtons_{nullptr};
  QPushButton *expand_{nullptr};
  rviz_common::RenderPanel *renderPanel_{nullptr};
  std::shared_ptr<RvizNode> rvizNode_;
  std::unique_ptr<rviz_common::VisualizationManager> manager_;
  rviz_common::Display *grid_{nullptr};
  rviz_common::Display *robot_{nullptr};
  rviz_common::Display *gripperAxes_{nullptr};
  rviz_common::Display *cameraAxes_{nullptr};
  rviz_common::Display *targetAxes_{nullptr};
  rviz_common::Display *targetMarker_{nullptr};
  std::vector<std::pair<QToolButton *, rviz_common::Tool *>> tools_;
  std::function<void(bool)> expandHandler_;
  QString fixedFrame_{"sgr532/base_link"};
  QString robotDescriptionTopic_{"/sgr532/robot_description"};
  QString cameraFrame_{"realsense"};
  QString targetFrame_{"calibration_target"};
  QString gripperFrame_{"hand_eye_gripper_pose"};
  QString targetMarkersTopic_{"/hand_eye_backend/target_markers"};
  QString cameraTransformSource_;
  bool cameraTransformActive_{false};
  bool gripperTransformActive_{false};
  bool targetTransformActive_{false};
  bool expanded_{false};
  bool active_{false};
  bool updateRunning_{false};
  bool initializationStarted_{false};
  bool ready_{false};
};

RvizWidget::RvizWidget(QWidget *parent)
    : QWidget(parent), impl_(std::make_unique<Impl>(*this)) {}

RvizWidget::~RvizWidget() = default;

void RvizWidget::configureFrames(
    const QString &fixedFrame, const QString &robotDescriptionTopic,
    const QString &cameraFrame, const QString &targetFrame,
    const QString &gripperFrame, const QString &targetMarkersTopic,
    const QString &cameraTransformSource, const bool cameraTransformActive,
    const bool gripperTransformActive, const bool targetTransformActive) {
  impl_->configureFrames(fixedFrame, robotDescriptionTopic, cameraFrame,
                         targetFrame, gripperFrame, targetMarkersTopic,
                         cameraTransformSource, cameraTransformActive,
                         gripperTransformActive, targetTransformActive);
}

void RvizWidget::initializeVisualization() { impl_->initialize(); }

void RvizWidget::setActive(const bool active) { impl_->setActive(active); }

void RvizWidget::setExpandHandler(std::function<void(bool)> handler) {
  impl_->setExpandHandler(std::move(handler));
}

void RvizWidget::setExpanded(const bool expanded) {
  impl_->setExpanded(expanded);
}

void RvizWidget::queueRender() { impl_->queueRender(); }

} // namespace hand_eye::app
