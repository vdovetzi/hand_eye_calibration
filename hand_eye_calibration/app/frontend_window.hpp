#pragma once

#include "app/frontend_backend_client.hpp"
#include "app/frontend_state.hpp"

#include <QByteArray>
#include <QMainWindow>
#include <QStringList>

#include <rclcpp/rclcpp.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace Ui {
class HandEyeFrontend;
}

class QTimer;

namespace hand_eye::app {

class CameraFeed;
class RvizWidget;

class FrontendWindow final : public QMainWindow {
public:
  explicit FrontendWindow(rclcpp::Node::SharedPtr node,
                          QWidget *parent = nullptr);
  ~FrontendWindow() override;

private:
  using DatasetMode = std::pair<bool, bool>; // overwrite, resume

  void configureVisuals();
  void configureSettingsLayout();
  void connectActions();
  void appendLog(const QString &message);
  void refreshTopics();
  void updatePatternStack(int index);

  [[nodiscard]] FrontendSettings settingsFromUi() const;
  void applySettings(const FrontendSettings &settings);
  void saveSettings();
  [[nodiscard]] bool validateSettings();
  [[nodiscard]] std::optional<DatasetMode> chooseDatasetMode();
  [[nodiscard]] bool pushParameters(bool overwrite = false,
                                    bool resume = false);

  void callTrigger(const QString &service, bool refreshStateAfter = true,
                   std::function<void()> onSuccess = {});
  void handleTrigger(TriggerResult result, bool refreshStateAfter,
                     const std::function<void()> &onSuccess);
  void pollBackendState();
  void applyBackendConfiguration(const BackendState &state);
  void renderBackendState(const BackendState &state);

  void startCameraPreview();
  void stopCameraPreview();
  void showWorkflowCamera();
  void applyWorkflowSplitterSizes();
  void showDonationDialog();
  void setRvizExpanded(bool expanded);

  rclcpp::Node::SharedPtr node_;
  std::unique_ptr<Ui::HandEyeFrontend> ui_;
  FrontendBackendClient backend_;
  std::unique_ptr<CameraFeed> cameraPreview_;
  std::unique_ptr<RvizWidget> rvizWidget_;
  QTimer *statePollTimer_{nullptr};
  QStringList renderedStatisticsLines_;
  QByteArray normalWindowGeometry_;
  QByteArray normalMainSplitterState_;
  QByteArray normalWorkflowSplitterState_;
  Qt::WindowStates normalWindowState_{Qt::WindowNoState};
  bool statsWasVisible_{true};
  bool logWasVisible_{true};
  bool progressWasVisible_{true};
  bool workflowActionsWereVisible_{true};
  bool tabBarWasVisible_{true};
  bool statusBarWasVisible_{true};
  bool backendConfigurationApplied_{false};
  bool statePollInFlight_{false};
  bool rvizExpanded_{false};
};

} // namespace hand_eye::app
