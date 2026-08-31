#include "app/frontend_window.hpp"

#include "app/camera_feed.hpp"
#include "app/rviz_widget.hpp"
#include "ui_hand_eye_frontend.h"

#include <QAbstractScrollArea>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QKeySequence>
#include <QLabel>
#include <QLayout>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QShortcut>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QStringList>
#include <QTabBar>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <initializer_list>
#include <utility>

namespace hand_eye::app {
namespace {

constexpr int kTypeRole = Qt::UserRole + 1;

QString selectedTopic(const QComboBox *combo) {
  const QVariant data = combo->currentData();
  return data.isValid() && !data.toString().trimmed().isEmpty()
             ? data.toString().trimmed()
             : combo->currentText().trimmed();
}

QString selectedTopicType(const QComboBox *combo) {
  return combo->currentData(kTypeRole).toString().trimmed();
}

void addTopic(QComboBox *combo, const TopicDescriptor &topic) {
  combo->addItem(topic.name, topic.name);
  const int index = combo->count() - 1;
  combo->setItemData(index, topic.type, kTypeRole);
  combo->setItemData(index, topic.type, Qt::ToolTipRole);
}

void setComboText(QComboBox *combo, const QString &text) {
  const int index = combo->findData(text);
  if (index >= 0) {
    combo->setCurrentIndex(index);
  } else {
    combo->setEditText(text);
  }
}

QString groupStyle(const QString &color) {
  return QString("QGroupBox { border: 1px solid #d8dde6; border-radius: 6px; "
                 "margin-top: 14px; padding-top: 10px; }"
                 "QGroupBox::title { subcontrol-origin: margin; left: 12px; "
                 "padding: 1px 8px; color: %1; font-weight: 700; "
                 "background: palette(window); }")
      .arg(color);
}

QString robotDescriptionTopic(const QString &baseFrame) {
  const QString frame = baseFrame.trimmed();
  const int separator = frame.indexOf('/');
  if (separator > 0) {
    return "/" + frame.left(separator) + "/robot_description";
  }
  return "/robot_description";
}

} // namespace

FrontendWindow::FrontendWindow(rclcpp::Node::SharedPtr node, QWidget *parent)
    : QMainWindow(parent), node_(std::move(node)),
      ui_(std::make_unique<Ui::HandEyeFrontend>()), backend_(node_, this) {
  ui_->setupUi(this);
  rvizWidget_ = std::make_unique<RvizWidget>(ui_->rvizContainer);
  ui_->rvizContainerLayout->addWidget(rvizWidget_.get());
  rvizWidget_->configureFrames("sgr532/base_link", "/sgr532/robot_description",
                               "realsense", "calibration_target",
                               "hand_eye_gripper_pose",
                               "/hand_eye_backend/target_markers");
  rvizWidget_->setActive(false);
  rvizWidget_->setExpandHandler(
      [this](const bool expanded) { setRvizExpanded(expanded); });
  auto *rvizFullscreenShortcut = new QShortcut(QKeySequence(Qt::Key_F11), this);
  rvizFullscreenShortcut->setContext(Qt::WindowShortcut);
  connect(rvizFullscreenShortcut, &QShortcut::activated, this, [this]() {
    if (rvizExpanded_ ||
        (ui_->mainStack->currentWidget() == ui_->workflowPage &&
         ui_->visualizationTabs->currentWidget() == ui_->rvizTab)) {
      setRvizExpanded(!rvizExpanded_);
    }
  });
  configureVisuals();
  connectActions();
  refreshTopics();

  QSettings settings("vdovetzi", "hand_eye_calibration");
  if (const auto saved = FrontendState::load(settings)) {
    applySettings(*saved);
  }
  updatePatternStack(ui_->patternTypeCombo->currentIndex());
  statePollTimer_ = new QTimer(this);
  statePollTimer_->setInterval(500);
  connect(statePollTimer_, &QTimer::timeout, this,
          [this]() { pollBackendState(); });
  statePollTimer_->start();
  QTimer::singleShot(100, this, [this]() { pollBackendState(); });
  if (node_->get_parameter("open_live_3d").as_bool()) {
    // Keep the historical parameter as an opt-in for entering the live
    // workflow directly, but always show the camera first. RViz is expensive
    // to initialize and is created only after the user selects its tab.
    showWorkflowCamera();
  }
  appendLog("Frontend started");
}

FrontendWindow::~FrontendWindow() { stopCameraPreview(); }

void FrontendWindow::configureVisuals() {
  ui_->mainStack->setCurrentWidget(ui_->settingsPage);
  ui_->visualizationTabs->setCurrentWidget(ui_->cameraTab);
  configureSettingsLayout();
  ui_->creditsLabel->setOpenExternalLinks(false);
  setMinimumSize(720, 500);
  ui_->mainStack->setMinimumHeight(360);
  ui_->mainSplitter->setChildrenCollapsible(false);
  ui_->mainSplitter->setCollapsible(0, false);
  ui_->mainSplitter->setCollapsible(1, false);
  ui_->mainSplitter->setHandleWidth(6);
  ui_->mainSplitter->setOpaqueResize(false);
  ui_->mainSplitter->setStretchFactor(0, 1);
  ui_->mainSplitter->setStretchFactor(1, 0);
  ui_->logPanel->setMinimumHeight(90);
  ui_->logPanel->setMaximumHeight(220);
  ui_->mainSplitter->setSizes({640, 110});
  ui_->workflowContentSplitter->setChildrenCollapsible(false);
  ui_->workflowContentSplitter->setCollapsible(0, false);
  ui_->workflowContentSplitter->setCollapsible(1, false);
  ui_->workflowContentSplitter->setHandleWidth(6);
  ui_->workflowContentSplitter->setOpaqueResize(false);
  ui_->workflowContentSplitter->setStretchFactor(0, 0);
  ui_->workflowContentSplitter->setStretchFactor(1, 1);
  ui_->statsGroup->setMinimumWidth(260);
  ui_->statsGroup->setMaximumWidth(360);
  ui_->statsGroup->setSizePolicy(QSizePolicy::MinimumExpanding,
                                 QSizePolicy::Expanding);
  ui_->visualizationTabs->setMinimumWidth(420);
  ui_->visualizationTabs->setSizePolicy(QSizePolicy::Expanding,
                                        QSizePolicy::Expanding);
  ui_->cameraPreviewLabel->setSizePolicy(QSizePolicy::Ignored,
                                         QSizePolicy::Ignored);
  ui_->statsList->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
  ui_->statsList->setUniformItemSizes(true);
  ui_->statsList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  ui_->statsList->setTextElideMode(Qt::ElideRight);
  ui_->workflowContentSplitter->setSizes({280, 900});
  ui_->progressBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  ui_->workflowActionsWidget->setSizePolicy(QSizePolicy::Expanding,
                                            QSizePolicy::Fixed);
  ui_->workflowLayout->setSpacing(4);
  ui_->workflowLayout->setStretch(0, 0);
  ui_->workflowLayout->setStretch(1, 0);
  ui_->workflowLayout->setStretch(2, 1);
  ui_->nextButton->setEnabled(false);
  ui_->nextButton->setToolTip("Waiting for the calibration backend...");
  ui_->calibrateButton->setEnabled(false);

  QFont titleFont = ui_->settingsTitleLabel->font();
  titleFont.setPointSize(titleFont.pointSize() + 8);
  titleFont.setBold(true);
  ui_->settingsTitleLabel->setFont(titleFont);
  ui_->creditsLabel->setTextFormat(Qt::RichText);
  ui_->creditsLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);
  ui_->creditsLabel->setStyleSheet("color: #666; font-size: 11px;");

  ui_->requiredGroup->setStyleSheet(groupStyle("#2f6fdd"));
  ui_->optionalGroup->setStyleSheet(groupStyle("#5b6470"));
  ui_->statisticsOptionsGroup->setStyleSheet(groupStyle("#147a5d"));
  ui_->statsGroup->setStyleSheet(groupStyle("#147a5d"));
  ui_->cameraGroup->setStyleSheet(groupStyle("#2f6fdd"));
  ui_->cameraPreviewLabel->setStyleSheet(
      "background: #111820; color: #d9e1ea; border-radius: 6px;");
  ui_->logGroup->setVisible(true);

  const std::initializer_list<std::pair<QWidget *, QString>> tooltips = {
      {ui_->armTopicLabel,
       "Arm Pose/Transform topic used for gripper samples."},
      {ui_->imageTopicLabel, "Camera topic used for calibration images."},
      {ui_->datasetPathLabel,
       "Folder containing captured images, poses.csv and sample metadata."},
      {ui_->patternTypeLabel, "Calibration target type."},
      {ui_->posesFormatLabel,
       "poses.csv encoding:\n1: x y z qx qy qz qw\n"
       "2: x y z qw qx qy qz\n3/4: roll pitch yaw [rad/deg]\n"
       "5/6: yaw pitch roll [rad/deg]"},
      {ui_->calibrationPathLabel, "Output calibration YAML file."},
      {ui_->intrinsicsPathLabel,
       "Camera K and D YAML. It may be omitted when CameraInfo is available."},
      {ui_->arucoDictLabel, "ArUco dictionary: 4x4, 5x5, 6x6, or 7x7."},
      {ui_->arucoIdLabel, "Marker id in the selected ArUco dictionary."},
      {ui_->arucoSizeLabel, "Physical marker size in meters."},
      {ui_->chessRowsLabel, "Number of inner chessboard corner rows."},
      {ui_->chessColsLabel, "Number of inner chessboard corner columns."},
      {ui_->chessCellSizeLabel, "Physical chessboard cell size in meters."},
      {ui_->charucoRowsLabel, "Number of ChArUco board rows."},
      {ui_->charucoColsLabel, "Number of ChArUco board columns."},
      {ui_->charucoCellSizeLabel, "Physical ChArUco cell size in meters."},
      {ui_->charucoMarkerSizeLabel, "Physical ChArUco marker size in meters."},
      {ui_->charucoDictLabel, "ArUco dictionary used by the ChArUco board."},
      {ui_->charucoIdLabel,
       "Legacy ChArUco marker id setting retained for compatibility."}};
  for (const auto &[widget, text] : tooltips) {
    widget->setToolTip(text);
  }
  ui_->eyeToHandCheck->setToolTip(
      "Enable eye-to-hand; disable for eye-in-hand calibration.");
  ui_->translationStatsCheck->setToolTip(
      "Show translation-space coverage of saved gripper poses.");
  ui_->rotationStatsCheck->setToolTip(
      "Show rotational coverage of saved gripper poses.");
  ui_->validationRmsStatsCheck->setToolTip(
      "Show target-lock consistency. Closer to zero is better.");
  updatePatternStack(0);
}

void FrontendWindow::configureSettingsLayout() {
  auto *scrollContents = new QWidget;
  scrollContents->setObjectName("settingsScrollContents");
  auto *contentLayout = new QVBoxLayout(scrollContents);
  contentLayout->setObjectName("settingsContentLayout");
  contentLayout->setContentsMargins(2, 2, 2, 2);
  contentLayout->setSpacing(ui_->settingsLayout->spacing());
  contentLayout->setSizeConstraint(QLayout::SetMinimumSize);

  const std::initializer_list<QWidget *> sections = {
      ui_->requiredGroup, ui_->patternStack, ui_->optionalGroup,
      ui_->statisticsOptionsGroup};
  for (QWidget *section : sections) {
    ui_->settingsLayout->removeWidget(section);
    section->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    contentLayout->addWidget(section);
  }
  ui_->settingsLayout->removeItem(ui_->settingsBottomSpacer);
  contentLayout->addItem(ui_->settingsBottomSpacer);

  auto *scrollArea = new QScrollArea(ui_->settingsPage);
  scrollArea->setObjectName("settingsScrollArea");
  scrollArea->setWidgetResizable(true);
  scrollArea->setFrameShape(QFrame::NoFrame);
  scrollArea->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
  scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  scrollArea->setMinimumHeight(240);
  scrollArea->setWidget(scrollContents);
  ui_->settingsLayout->insertWidget(1, scrollArea, 1);

  ui_->creditsLabel->setMinimumWidth(0);
  ui_->creditsLabel->setSizePolicy(QSizePolicy::Ignored,
                                   QSizePolicy::Preferred);

  QFont readableFont = font();
  if (readableFont.pointSizeF() < 10.0) {
    readableFont.setPointSizeF(10.0);
  }
  scrollContents->setFont(readableFont);
  ui_->statsGroup->setFont(readableFont);
  ui_->statsList->setFont(readableFont);
  ui_->requiredGroup->setMinimumHeight(
      ui_->requiredGroup->minimumSizeHint().height());
  ui_->statisticsOptionsGroup->setMinimumHeight(
      ui_->statisticsOptionsGroup->minimumSizeHint().height());
}

void FrontendWindow::connectActions() {
  connect(ui_->patternTypeCombo,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [this](const int index) { updatePatternStack(index); });
  connect(ui_->visualizationTabs, &QTabWidget::currentChanged, this,
          [this](const int) {
            if (ui_->visualizationTabs->currentWidget() == ui_->cameraTab) {
              rvizWidget_->setActive(false);
              if (ui_->mainStack->currentWidget() == ui_->workflowPage) {
                startCameraPreview();
              }
              return;
            }
            // Avoid decoding and repainting a hidden camera label while RViz
            // is active. The backend camera subscription is independent.
            stopCameraPreview();
            rvizWidget_->setActive(true);
            QTimer::singleShot(100, this, [this]() {
              if (ui_->visualizationTabs->currentWidget() == ui_->rvizTab &&
                  ui_->rvizTab->isVisible()) {
                rvizWidget_->initializeVisualization();
              }
            });
          });
  connect(ui_->refreshTopicsButton, &QPushButton::clicked, this,
          [this]() { refreshTopics(); });
  connect(ui_->creditsLabel, &QLabel::linkActivated, this,
          [this](const QString &link) {
            if (link == "donate://coffee") {
              showDonationDialog();
            } else {
              QDesktopServices::openUrl(QUrl(link));
            }
          });
  connect(ui_->datasetBrowseButton, &QPushButton::clicked, this, [this]() {
    const QString path =
        QFileDialog::getExistingDirectory(this, "Select dataset folder");
    if (!path.isEmpty()) {
      ui_->datasetPathEdit->setText(path);
    }
  });
  connect(ui_->calibrationBrowseButton, &QPushButton::clicked, this, [this]() {
    const QString path = QFileDialog::getSaveFileName(
        this, "Select calibration YAML", settingsFromUi().calibrationPath,
        "YAML files (*.yaml *.yml)");
    if (!path.isEmpty()) {
      ui_->calibrationPathEdit->setText(path);
    }
  });
  connect(ui_->intrinsicsBrowseButton, &QPushButton::clicked, this, [this]() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Select intrinsics YAML", {}, "YAML files (*.yaml *.yml)");
    if (!path.isEmpty()) {
      ui_->intrinsicsPathEdit->setText(path);
    }
  });
  connect(ui_->saveSettingsButton, &QPushButton::clicked, this, [this]() {
    saveSettings();
    appendLog("Settings saved");
  });
  connect(ui_->nextButton, &QPushButton::clicked, this, [this]() {
    if (!validateSettings()) {
      return;
    }
    const auto mode = chooseDatasetMode();
    if (!mode) {
      appendLog("Dataset selection canceled");
      return;
    }
    saveSettings();
    if (!pushParameters(mode->first, mode->second)) {
      return;
    }
    showWorkflowCamera();
    callTrigger("configure", true, [this]() {
      callTrigger("prepare_dataset", true,
                  [this]() { callTrigger("start_collection"); });
    });
  });
  connect(ui_->backToSettingsButton, &QPushButton::clicked, this, [this]() {
    if (rvizExpanded_) {
      setRvizExpanded(false);
    }
    rvizWidget_->setActive(false);
    stopCameraPreview();
    ui_->mainStack->setCurrentWidget(ui_->settingsPage);
  });
  connect(ui_->captureSampleButton, &QPushButton::clicked, this, [this]() {
    if (pushParameters()) {
      callTrigger("capture_sample");
    }
  });
  connect(ui_->removeSampleButton, &QPushButton::clicked, this,
          [this]() { callTrigger("remove_sample"); });
  connect(ui_->calibrateButton, &QPushButton::clicked, this, [this]() {
    if (!ui_->calibrateButton->isEnabled()) {
      appendLog("Dataset quality is not ready for calibration yet");
    } else if (pushParameters()) {
      callTrigger("run_calibration");
    }
  });
}

void FrontendWindow::appendLog(const QString &message) {
  const QString timestamp =
      QDateTime::currentDateTime().toString("HH:mm:ss.zzz");
  ui_->logText->appendPlainText(QString("[%1] %2").arg(timestamp, message));
}

void FrontendWindow::refreshTopics() {
  const QString oldArm = selectedTopic(ui_->armTopicCombo);
  const QString oldImage = selectedTopic(ui_->imageTopicCombo);
  ui_->armTopicCombo->clear();
  ui_->imageTopicCombo->clear();
  ui_->armTopicCombo->addItem("", "");
  ui_->imageTopicCombo->addItem("", "");
  const TopicGraph graph = backend_.discoverTopics();
  for (const auto &topic : graph.armTopics) {
    addTopic(ui_->armTopicCombo, topic);
  }
  for (const auto &topic : graph.imageTopics) {
    addTopic(ui_->imageTopicCombo, topic);
  }
  setComboText(ui_->armTopicCombo, oldArm);
  setComboText(ui_->imageTopicCombo, oldImage);
  appendLog("Topics refreshed");
}

void FrontendWindow::updatePatternStack(const int index) {
  ui_->patternStack->setCurrentIndex(index);
  ui_->patternStack->setVisible(index > 0);
  const QWidget *page =
      index > 0 ? ui_->patternStack->currentWidget() : nullptr;
  ui_->patternStack->setFixedHeight(
      page ? std::max(1, page->sizeHint().height() + 8) : 0);
  ui_->patternStack->updateGeometry();
}

FrontendSettings FrontendWindow::settingsFromUi() const {
  FrontendSettings value;
  value.armTopic = selectedTopic(ui_->armTopicCombo);
  value.imageTopic = selectedTopic(ui_->imageTopicCombo);
  value.imageTopicType = selectedTopicType(ui_->imageTopicCombo);
  value.datasetPath = ui_->datasetPathEdit->text().trimmed();
  value.calibrationPath = ui_->calibrationPathEdit->text().trimmed();
  if (value.calibrationPath.isEmpty()) {
    value.calibrationPath = "calibration.yaml";
  }
  value.intrinsicsPath = ui_->intrinsicsPathEdit->text().trimmed();
  value.pattern.type = ui_->patternTypeCombo->currentIndex();
  value.pattern.arucoDictionary = ui_->arucoDictSpin->value();
  value.pattern.arucoId = ui_->arucoIdSpin->value();
  value.pattern.arucoMarkerSize = ui_->arucoMarkerSizeSpin->value();
  value.pattern.chessRows = ui_->chessRowsSpin->value();
  value.pattern.chessColumns = ui_->chessColsSpin->value();
  value.pattern.chessCellSize = ui_->chessCellSizeSpin->value();
  value.pattern.charucoRows = ui_->charucoRowsSpin->value();
  value.pattern.charucoColumns = ui_->charucoColsSpin->value();
  value.pattern.charucoCellSize = ui_->charucoCellSizeSpin->value();
  value.pattern.charucoMarkerSize = ui_->charucoMarkerSizeSpin->value();
  value.pattern.charucoDictionary = ui_->charucoDictSpin->value();
  value.pattern.charucoId = ui_->charucoIdSpin->value();
  value.posesFormat = ui_->posesFormatSpin->value();
  value.eyeToHand = ui_->eyeToHandCheck->isChecked();
  value.translationStatistics = ui_->translationStatsCheck->isChecked();
  value.rotationStatistics = ui_->rotationStatsCheck->isChecked();
  value.validationRmsStatistics = ui_->validationRmsStatsCheck->isChecked();
  return value;
}

void FrontendWindow::applySettings(const FrontendSettings &value) {
  setComboText(ui_->armTopicCombo, value.armTopic);
  setComboText(ui_->imageTopicCombo, value.imageTopic);
  ui_->datasetPathEdit->setText(value.datasetPath);
  ui_->calibrationPathEdit->setText(value.calibrationPath);
  ui_->intrinsicsPathEdit->setText(value.intrinsicsPath);
  ui_->patternTypeCombo->setCurrentIndex(value.pattern.type);
  ui_->arucoDictSpin->setValue(value.pattern.arucoDictionary);
  ui_->arucoIdSpin->setValue(value.pattern.arucoId);
  ui_->arucoMarkerSizeSpin->setValue(value.pattern.arucoMarkerSize);
  ui_->chessRowsSpin->setValue(value.pattern.chessRows);
  ui_->chessColsSpin->setValue(value.pattern.chessColumns);
  ui_->chessCellSizeSpin->setValue(value.pattern.chessCellSize);
  ui_->charucoRowsSpin->setValue(value.pattern.charucoRows);
  ui_->charucoColsSpin->setValue(value.pattern.charucoColumns);
  ui_->charucoCellSizeSpin->setValue(value.pattern.charucoCellSize);
  ui_->charucoMarkerSizeSpin->setValue(value.pattern.charucoMarkerSize);
  ui_->charucoDictSpin->setValue(value.pattern.charucoDictionary);
  ui_->charucoIdSpin->setValue(value.pattern.charucoId);
  ui_->posesFormatSpin->setValue(value.posesFormat);
  ui_->eyeToHandCheck->setChecked(value.eyeToHand);
  ui_->translationStatsCheck->setChecked(value.translationStatistics);
  ui_->rotationStatsCheck->setChecked(value.rotationStatistics);
  ui_->validationRmsStatsCheck->setChecked(value.validationRmsStatistics);
}

void FrontendWindow::saveSettings() {
  QSettings settings("vdovetzi", "hand_eye_calibration");
  FrontendState::save(settings, settingsFromUi());
}

bool FrontendWindow::validateSettings() {
  const FrontendSettings value = settingsFromUi();
  QStringList missing;
  if (value.armTopic.isEmpty()) {
    missing << "arm topic";
  }
  if (value.imageTopic.isEmpty()) {
    missing << "camera topic";
  }
  if (value.datasetPath.isEmpty()) {
    missing << "dataset output folder";
  }
  if (value.calibrationPath.isEmpty()) {
    missing << "calibration output path";
  }
  if (value.pattern.type == 0) {
    missing << "pattern type";
  }
  if (missing.isEmpty()) {
    return true;
  }
  QMessageBox::warning(this, "Missing settings",
                       "Fill required fields: " + missing.join(", "));
  return false;
}

std::optional<FrontendWindow::DatasetMode> FrontendWindow::chooseDatasetMode() {
  const QString path = settingsFromUi().datasetPath;
  const QFileInfo info(path);
  if (info.exists() && !info.isDir()) {
    const auto answer = QMessageBox::warning(
        this, "Dataset path is not a folder",
        "The selected path is not a folder. Replace it with a new dataset?",
        QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    return answer == QMessageBox::Yes
               ? std::optional<DatasetMode>(DatasetMode{true, false})
               : std::nullopt;
  }
  const QDir directory(path);
  if (!directory.exists() ||
      directory
          .entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries | QDir::Hidden)
          .isEmpty()) {
    // QFileDialog commonly creates the selected directory. Mark an existing
    // empty directory for replacement so the repository can initialize its
    // images/, poses.csv and samples.tsv contract there.
    return DatasetMode{directory.exists(), false};
  }

  QMessageBox box(this);
  box.setWindowTitle("Dataset folder is not empty");
  box.setText("The selected dataset folder already exists and is not empty.");
  box.setInformativeText(
      "Continue the existing dataset, start over by removing it, or cancel?");
  auto *resume = box.addButton("Continue Dataset", QMessageBox::AcceptRole);
  auto *overwrite = box.addButton("Start Over", QMessageBox::DestructiveRole);
  box.addButton(QMessageBox::Cancel);
  box.setDefaultButton(resume);
  box.exec();
  if (box.clickedButton() == resume) {
    return DatasetMode{false, true};
  }
  if (box.clickedButton() == overwrite) {
    return DatasetMode{true, false};
  }
  return std::nullopt;
}

bool FrontendWindow::pushParameters(const bool overwrite, const bool resume) {
  const ParameterUpdateResult result =
      backend_.pushParameters(settingsFromUi(), overwrite, resume);
  appendLog(result.message);
  return result.success;
}

void FrontendWindow::callTrigger(const QString &service,
                                 const bool refreshStateAfter,
                                 std::function<void()> onSuccess) {
  appendLog("Calling /hand_eye_backend/" + service);
  backend_.callTrigger(
      service, [this, refreshStateAfter,
                onSuccess = std::move(onSuccess)](TriggerResult result) {
        handleTrigger(std::move(result), refreshStateAfter, onSuccess);
      });
}

void FrontendWindow::handleTrigger(TriggerResult result,
                                   const bool refreshStateAfter,
                                   const std::function<void()> &onSuccess) {
  const QString message =
      result.service == "get_state" && result.success
          ? "State refreshed"
          : FrontendState::humanizeBackendMessage(result.message);
  appendLog((result.success ? "[ok] " : "[failed] ") + result.service + ": " +
            message);
  if (result.service == "get_state" && result.success) {
    if (const auto state = FrontendState::parseBackendState(result.message)) {
      renderBackendState(*state);
    }
  } else if (result.success && onSuccess) {
    onSuccess();
  } else if (refreshStateAfter) {
    pollBackendState();
  }
}

void FrontendWindow::pollBackendState() {
  if (!rclcpp::ok(node_->get_node_base_interface()->get_context())) {
    statePollInFlight_ = false;
    statePollTimer_->stop();
    return;
  }
  if (statePollInFlight_) {
    return;
  }
  statePollInFlight_ = true;
  try {
    backend_.callTrigger("get_state", [this](TriggerResult result) {
      statePollInFlight_ = false;
      if (!result.success) {
        return;
      }
      if (const auto state = FrontendState::parseBackendState(result.message)) {
        renderBackendState(*state);
      }
    });
  } catch (const std::exception &error) {
    statePollInFlight_ = false;
    statePollTimer_->stop();
    appendLog("Backend state polling stopped: " +
              QString::fromUtf8(error.what()));
  }
}

void FrontendWindow::applyBackendConfiguration(const BackendState &state) {
  if (backendConfigurationApplied_) {
    return;
  }
  const auto pattern = FrontendState::parsePattern(state.patternInfo);
  const bool hasLaunchConfiguration = !state.armTopic.trimmed().isEmpty() ||
                                      !state.imageTopic.trimmed().isEmpty() ||
                                      pattern.has_value() ||
                                      !state.intrinsicsPath.trimmed().isEmpty();
  if (!hasLaunchConfiguration) {
    return;
  }

  FrontendSettings settings = settingsFromUi();
  settings.armTopic = state.armTopic;
  settings.imageTopic = state.imageTopic;
  settings.datasetPath = state.datasetPath;
  settings.calibrationPath = state.calibrationPath;
  settings.intrinsicsPath = state.intrinsicsPath;
  settings.posesFormat = state.posesFormat;
  settings.eyeToHand = state.eyeToHand;
  if (pattern) {
    settings.pattern = *pattern;
  }
  applySettings(settings);
  backendConfigurationApplied_ = true;
  appendLog("Loaded calibration settings from backend launch configuration");
  if (ui_->mainStack->currentWidget() == ui_->workflowPage &&
      ui_->visualizationTabs->currentWidget() == ui_->cameraTab) {
    startCameraPreview();
  }
}

void FrontendWindow::renderBackendState(const BackendState &state) {
  applyBackendConfiguration(state);
  ui_->nextButton->setEnabled(true);
  ui_->nextButton->setToolTip("Configure the backend and start collection.");
  const FrontendViewState view =
      FrontendState::makeViewState(state, settingsFromUi());
  ui_->progressBar->setValue(view.progressValue);
  ui_->calibrateButton->setEnabled(view.calibrationReady);
  ui_->calibrateButton->setToolTip(
      view.calibrationReady
          ? "Run hand-eye calibration."
          : "Calibration is enabled when dataset quality reaches 100%.");
  if (view.statisticsLines != renderedStatisticsLines_) {
    ui_->statsList->setUpdatesEnabled(false);
    for (int index = 0; index < view.statisticsLines.size(); ++index) {
      const QString &line = view.statisticsLines.at(index);
      QListWidgetItem *item = ui_->statsList->item(index);
      if (!item) {
        item = new QListWidgetItem(ui_->statsList);
      }
      if (item->text() != line) {
        item->setText(line);
      }
      item->setToolTip(line);
    }
    while (ui_->statsList->count() > view.statisticsLines.size()) {
      delete ui_->statsList->takeItem(ui_->statsList->count() - 1);
    }
    ui_->statsList->setUpdatesEnabled(true);
    ui_->statsList->viewport()->update();
    renderedStatisticsLines_ = view.statisticsLines;
  }
  QString cameraTransformStatus = state.cameraTfSource.trimmed();
  if (cameraTransformStatus.compare("none", Qt::CaseInsensitive) == 0) {
    cameraTransformStatus.clear();
  }
  if (cameraTransformStatus.isEmpty() && state.initialCameraTfConfigured) {
    cameraTransformStatus = "initial";
  }
  if (!cameraTransformStatus.isEmpty() &&
      !state.cameraTfParentFrame.trimmed().isEmpty()) {
    cameraTransformStatus =
        QString("%1 from %2")
            .arg(cameraTransformStatus, state.cameraTfParentFrame);
  }
  rvizWidget_->configureFrames(
      state.robotBaseFrame, robotDescriptionTopic(state.robotBaseFrame),
      state.cameraFrame, state.targetFrame, state.gripperVisualizationFrame,
      state.targetMarkersTopic, cameraTransformStatus, state.cameraTfActive,
      state.gripperTfActive, state.targetTfVisible);
}

void FrontendWindow::setRvizExpanded(const bool expanded) {
  if (expanded == rvizExpanded_) {
    return;
  }

  if (expanded) {
    if (ui_->mainStack->currentWidget() != ui_->workflowPage ||
        ui_->visualizationTabs->currentWidget() != ui_->rvizTab) {
      return;
    }
    ui_->mainStack->setCurrentWidget(ui_->workflowPage);
    rvizWidget_->initializeVisualization();

    normalWindowGeometry_ = saveGeometry();
    normalWindowState_ = windowState();
    normalMainSplitterState_ = ui_->mainSplitter->saveState();
    normalWorkflowSplitterState_ = ui_->workflowContentSplitter->saveState();
    statsWasVisible_ = !ui_->statsGroup->isHidden();
    logWasVisible_ = !ui_->logPanel->isHidden();
    progressWasVisible_ = !ui_->progressBar->isHidden();
    workflowActionsWereVisible_ = !ui_->workflowActionsWidget->isHidden();
    tabBarWasVisible_ = !ui_->visualizationTabs->tabBar()->isHidden();
    statusBarWasVisible_ = !statusBar()->isHidden();

    rvizExpanded_ = true;
    ui_->statsGroup->hide();
    ui_->logPanel->hide();
    ui_->progressBar->hide();
    ui_->workflowActionsWidget->hide();
    ui_->visualizationTabs->tabBar()->hide();
    statusBar()->hide();
    rvizWidget_->setExpanded(true);
    showFullScreen();
  } else {
    rvizExpanded_ = false;
    showNormal();
    if (!normalWindowGeometry_.isEmpty()) {
      restoreGeometry(normalWindowGeometry_);
    }
    setWindowState(normalWindowState_ & ~Qt::WindowFullScreen);

    ui_->statsGroup->setVisible(statsWasVisible_);
    ui_->logPanel->setVisible(logWasVisible_);
    ui_->progressBar->setVisible(progressWasVisible_);
    ui_->workflowActionsWidget->setVisible(workflowActionsWereVisible_);
    ui_->visualizationTabs->tabBar()->setVisible(tabBarWasVisible_);
    statusBar()->setVisible(statusBarWasVisible_);
    if (!normalMainSplitterState_.isEmpty()) {
      ui_->mainSplitter->restoreState(normalMainSplitterState_);
    }
    if (!normalWorkflowSplitterState_.isEmpty()) {
      ui_->workflowContentSplitter->restoreState(normalWorkflowSplitterState_);
    }
    rvizWidget_->setExpanded(false);
  }

  QTimer::singleShot(0, this, [this]() { rvizWidget_->queueRender(); });
}

void FrontendWindow::startCameraPreview() {
  const FrontendSettings settings = settingsFromUi();
  if (settings.imageTopic.isEmpty()) {
    ui_->cameraPreviewLabel->setText("Choose a camera topic in Settings.");
    return;
  }
  stopCameraPreview();
  ui_->cameraPreviewLabel->setText("Waiting for camera frames...");
  cameraPreview_ = std::make_unique<CameraFeed>(
      node_, settings.imageTopic.toStdString(),
      settings.imageTopicType.toStdString(), ui_->cameraPreviewLabel,
      FrontendState::serializePattern(settings.pattern).toStdString());
}

void FrontendWindow::showWorkflowCamera() {
  rvizWidget_->setActive(false);
  ui_->visualizationTabs->setCurrentWidget(ui_->cameraTab);
  ui_->mainStack->setCurrentWidget(ui_->workflowPage);
  startCameraPreview();
  QTimer::singleShot(0, this, [this]() { applyWorkflowSplitterSizes(); });
}

void FrontendWindow::applyWorkflowSplitterSizes() {
  const int contentWidth = ui_->workflowContentSplitter->width();
  if (contentWidth > 0) {
    const int statisticsWidth = std::clamp(contentWidth / 4, 260, 320);
    ui_->workflowContentSplitter->setSizes(
        {statisticsWidth,
         std::max(420, contentWidth - statisticsWidth -
                           ui_->workflowContentSplitter->handleWidth())});
  }

  const int contentHeight = ui_->mainSplitter->height();
  if (contentHeight > 0) {
    const int logHeight = std::clamp(contentHeight / 7, 90, 130);
    ui_->mainSplitter->setSizes(
        {std::max(360,
                  contentHeight - logHeight - ui_->mainSplitter->handleWidth()),
         logHeight});
  }
}

void FrontendWindow::stopCameraPreview() {
  cameraPreview_.reset();
  if (ui_) {
    ui_->cameraPreviewLabel->setPixmap({});
    ui_->cameraPreviewLabel->setText("Camera stream will appear after setup.");
  }
}

void FrontendWindow::showDonationDialog() {
  QDialog dialog(this);
  dialog.setWindowTitle("Buy me a coffee");
  auto *layout = new QVBoxLayout(&dialog);
  auto *image = new QLabel(&dialog);
  image->setAlignment(Qt::AlignCenter);
  const QPixmap donation(":/hand_eye_calibration/donation_usdt_erc20.png");
  image->setPixmap(
      donation.scaled(520, 640, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  layout->addWidget(image);
  auto *close = new QPushButton("Close", &dialog);
  connect(close, &QPushButton::clicked, &dialog, &QDialog::accept);
  layout->addWidget(close, 0, Qt::AlignRight);
  dialog.exec();
}

} // namespace hand_eye::app
