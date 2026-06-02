#include "ui_hand_eye_frontend.h"

#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QMetaObject>
#include <QPixmap>
#include <QPushButton>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidget>

#include <cv_bridge/cv_bridge.hpp>
#include <hand_eye_calibration/pattern_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using Trigger = std_srvs::srv::Trigger;
using namespace std::chrono_literals;

constexpr const char *kImageType = "sensor_msgs/msg/Image";
constexpr const char *kCompressedImageType = "sensor_msgs/msg/CompressedImage";
constexpr const char *kBackendNode = "/hand_eye_backend";

bool isSupportedArmType(const std::string &type) {
  return type == "geometry_msgs/msg/Pose" ||
         type == "geometry_msgs/msg/PoseStamped" ||
         type == "geometry_msgs/msg/Transform" ||
         type == "geometry_msgs/msg/TransformStamped" ||
         type == "sensor_msgs/msg/JointState";
}

std::string serviceName(const std::string &backendNode,
                        const std::string &service) {
  return backendNode + "/" + service;
}

bool topicHasType(const std::vector<std::string> &types,
                  const std::string &expectedType) {
  return std::ranges::find(types, expectedType) != types.end();
}

QString baseImageTopic(const QString &topic) {
  constexpr const char *suffix = "/compressed";
  return topic.endsWith(suffix) ? topic.left(topic.size() - std::strlen(suffix))
                                : topic;
}

void appendLog(Ui::HandEyeFrontend &ui, const QString &message) {
  const QString timestamp =
      QDateTime::currentDateTime().toString("HH:mm:ss.zzz");
  ui.logText->appendPlainText(QString("[%1] %2").arg(timestamp, message));
}

QString selectedTopic(const QComboBox *combo) {
  const QVariant data = combo->currentData();
  if (data.isValid() && !data.toString().trimmed().isEmpty()) {
    return data.toString().trimmed();
  }
  return combo->currentText().trimmed();
}

QString selectedTopicType(const QComboBox *combo) {
  return combo->currentData(Qt::UserRole + 1).toString().trimmed();
}

QString userMessage(const std::string &message) {
  QString text = QString::fromStdString(message).trimmed();
  const int jsonStart = text.indexOf('{');
  if (jsonStart > 0) {
    text = text.left(jsonStart).trimmed();
  }
  text.replace("dataset_path", "dataset folder");
  text.replace("image_topic", "camera topic");
  text.replace("arm_topic", "arm topic");
  text.replace("pattern_info", "pattern settings");
  text.replace("intrinsics_path", "intrinsics YAML");
  text.replace("resume_dataset=true", "Continue Dataset");
  text.replace("overwrite_dataset=true", "Start Over");
  text.replace("last_validation_rms", "target lock RMS");
  text.replace("validation_rms", "target lock RMS");
  text.replace("target_lock_rms", "Target lock RMS");
  return text;
}

QString calibrationPath(Ui::HandEyeFrontend &ui) {
  const QString path = ui.calibrationPathEdit->text().trimmed();
  return path.isEmpty() ? QString("calibration.yaml") : path;
}

void addTopicItem(QComboBox *combo, const QString &topic, const QString &type) {
  combo->addItem(topic, topic);
  const int index = combo->count() - 1;
  combo->setItemData(index, type, Qt::UserRole + 1);
  combo->setItemData(index, type, Qt::ToolTipRole);
}

void setComboText(QComboBox *combo, const QString &text) {
  const int index = combo->findData(text);
  if (index >= 0) {
    combo->setCurrentIndex(index);
  } else {
    combo->setEditText(text);
  }
}

QString patternInfo(Ui::HandEyeFrontend &ui) {
  switch (ui.patternTypeCombo->currentIndex()) {
  case 1:
    return QString("1 %1 %2 %3")
        .arg(ui.arucoDictSpin->value())
        .arg(ui.arucoIdSpin->value())
        .arg(ui.arucoMarkerSizeSpin->value(), 0, 'f', 4);
  case 2:
    return QString("2 %1 %2 %3")
        .arg(ui.chessRowsSpin->value())
        .arg(ui.chessColsSpin->value())
        .arg(ui.chessCellSizeSpin->value(), 0, 'f', 4);
  case 3:
    return QString("3 %1 %2 %3 %4 %5 %6")
        .arg(ui.charucoRowsSpin->value())
        .arg(ui.charucoColsSpin->value())
        .arg(ui.charucoCellSizeSpin->value(), 0, 'f', 4)
        .arg(ui.charucoMarkerSizeSpin->value(), 0, 'f', 4)
        .arg(ui.charucoDictSpin->value())
        .arg(ui.charucoIdSpin->value());
  default:
    return {};
  }
}

std::vector<rclcpp::Parameter> collectParameters(Ui::HandEyeFrontend &ui,
                                                 bool overwriteDataset,
                                                 bool resumeDataset) {
  return {
      rclcpp::Parameter("arm_topic",
                        selectedTopic(ui.armTopicCombo).toStdString()),
      rclcpp::Parameter("image_topic",
                        selectedTopic(ui.imageTopicCombo).toStdString()),
      rclcpp::Parameter("dataset_path",
                        ui.datasetPathEdit->text().toStdString()),
      rclcpp::Parameter("calibration_path", calibrationPath(ui).toStdString()),
      rclcpp::Parameter("pattern_info", patternInfo(ui).toStdString()),
      rclcpp::Parameter("poses_format", ui.posesFormatSpin->value()),
      rclcpp::Parameter("eye_to_hand", ui.eyeToHandCheck->isChecked()),
      rclcpp::Parameter("intrinsics_path",
                        ui.intrinsicsPathEdit->text().toStdString()),
      rclcpp::Parameter("require_pattern_detection", true),
      rclcpp::Parameter("overwrite_dataset", overwriteDataset),
      rclcpp::Parameter("resume_dataset", resumeDataset),
      rclcpp::Parameter("min_samples", 3),
      rclcpp::Parameter("stat_translation_spread",
                        ui.translationStatsCheck->isChecked()),
      rclcpp::Parameter("stat_rotation_spread",
                        ui.rotationStatsCheck->isChecked()),
      rclcpp::Parameter("stat_validation_rms",
                        ui.validationRmsStatsCheck->isChecked()),
  };
}

void saveSettings(Ui::HandEyeFrontend &ui) {
  QSettings settings("vdovetzi", "hand_eye_calibration");
  settings.setValue("saved", true);
  settings.setValue("arm_topic", selectedTopic(ui.armTopicCombo));
  settings.setValue("image_topic", selectedTopic(ui.imageTopicCombo));
  settings.setValue("dataset_path", ui.datasetPathEdit->text());
  settings.setValue("calibration_path", calibrationPath(ui));
  settings.setValue("intrinsics_path", ui.intrinsicsPathEdit->text());
  settings.setValue("pattern_type", ui.patternTypeCombo->currentIndex());
  settings.setValue("poses_format", ui.posesFormatSpin->value());
  settings.setValue("eye_to_hand", ui.eyeToHandCheck->isChecked());
  settings.setValue("translation_stats", ui.translationStatsCheck->isChecked());
  settings.setValue("rotation_stats", ui.rotationStatsCheck->isChecked());
  settings.setValue("validation_rms_stats",
                    ui.validationRmsStatsCheck->isChecked());
  settings.setValue("aruco_dict", ui.arucoDictSpin->value());
  settings.setValue("aruco_id", ui.arucoIdSpin->value());
  settings.setValue("aruco_marker_size", ui.arucoMarkerSizeSpin->value());
  settings.setValue("chess_rows", ui.chessRowsSpin->value());
  settings.setValue("chess_cols", ui.chessColsSpin->value());
  settings.setValue("chess_cell_size", ui.chessCellSizeSpin->value());
  settings.setValue("charuco_rows", ui.charucoRowsSpin->value());
  settings.setValue("charuco_cols", ui.charucoColsSpin->value());
  settings.setValue("charuco_cell_size", ui.charucoCellSizeSpin->value());
  settings.setValue("charuco_marker_size", ui.charucoMarkerSizeSpin->value());
  settings.setValue("charuco_dict", ui.charucoDictSpin->value());
  settings.setValue("charuco_id", ui.charucoIdSpin->value());
}

void loadSettings(Ui::HandEyeFrontend &ui) {
  QSettings settings("vdovetzi", "hand_eye_calibration");
  if (!settings.value("saved", false).toBool()) {
    return;
  }
  setComboText(ui.armTopicCombo, settings.value("arm_topic").toString());
  setComboText(ui.imageTopicCombo, settings.value("image_topic").toString());
  ui.datasetPathEdit->setText(settings.value("dataset_path").toString());
  const QString savedCalibrationPath =
      settings.value("calibration_path", "calibration.yaml")
          .toString()
          .trimmed();
  ui.calibrationPathEdit->setText(savedCalibrationPath.isEmpty()
                                      ? "calibration.yaml"
                                      : savedCalibrationPath);
  ui.intrinsicsPathEdit->setText(settings.value("intrinsics_path").toString());
  ui.patternTypeCombo->setCurrentIndex(
      settings.value("pattern_type", 0).toInt());
  ui.posesFormatSpin->setValue(settings.value("poses_format", 1).toInt());
  ui.eyeToHandCheck->setChecked(settings.value("eye_to_hand", false).toBool());
  ui.translationStatsCheck->setChecked(
      settings.value("translation_stats", true).toBool());
  ui.rotationStatsCheck->setChecked(
      settings.value("rotation_stats", true).toBool());
  ui.validationRmsStatsCheck->setChecked(
      settings.value("validation_rms_stats", false).toBool());
  ui.arucoDictSpin->setValue(settings.value("aruco_dict", 4).toInt());
  ui.arucoIdSpin->setValue(settings.value("aruco_id", 0).toInt());
  ui.arucoMarkerSizeSpin->setValue(
      settings.value("aruco_marker_size", 0.0).toDouble());
  ui.chessRowsSpin->setValue(settings.value("chess_rows", 1).toInt());
  ui.chessColsSpin->setValue(settings.value("chess_cols", 1).toInt());
  ui.chessCellSizeSpin->setValue(
      settings.value("chess_cell_size", 0.0).toDouble());
  ui.charucoRowsSpin->setValue(settings.value("charuco_rows", 1).toInt());
  ui.charucoColsSpin->setValue(settings.value("charuco_cols", 1).toInt());
  ui.charucoCellSizeSpin->setValue(
      settings.value("charuco_cell_size", 0.0).toDouble());
  ui.charucoMarkerSizeSpin->setValue(
      settings.value("charuco_marker_size", 0.0).toDouble());
  ui.charucoDictSpin->setValue(settings.value("charuco_dict", 4).toInt());
  ui.charucoIdSpin->setValue(settings.value("charuco_id", 0).toInt());
}

void refreshGraph(const rclcpp::Node::SharedPtr &node,
                  Ui::HandEyeFrontend &ui) {
  const QString oldArmTopic = selectedTopic(ui.armTopicCombo);
  const QString oldImageTopic = selectedTopic(ui.imageTopicCombo);

  ui.armTopicCombo->clear();
  ui.imageTopicCombo->clear();
  ui.armTopicCombo->addItem("", "");
  ui.imageTopicCombo->addItem("", "");

  const auto topics = node->get_topic_names_and_types();
  std::map<QString, QString> imageTopics;
  for (const auto &[name, types] : topics) {
    const QString topic = QString::fromStdString(name);
    const bool hasCompressed = topicHasType(types, kCompressedImageType);
    const bool hasImage = topicHasType(types, kImageType);
    if (hasCompressed) {
      imageTopics[baseImageTopic(topic)] =
          QString::fromStdString(kCompressedImageType);
    } else if (hasImage && !imageTopics.contains(topic)) {
      imageTopics[topic] = QString::fromStdString(kImageType);
    }

    for (const std::string &type : types) {
      if (isSupportedArmType(type)) {
        addTopicItem(ui.armTopicCombo, topic, QString::fromStdString(type));
        break;
      }
    }
  }

  for (const auto &[topic, type] : imageTopics) {
    addTopicItem(ui.imageTopicCombo, topic, type);
  }

  setComboText(ui.armTopicCombo, oldArmTopic);
  setComboText(ui.imageTopicCombo, oldImageTopic);
  appendLog(ui, "Topics refreshed");
}

bool pushParameters(const rclcpp::Node::SharedPtr &node,
                    Ui::HandEyeFrontend &ui,
                    const bool overwriteDataset = false,
                    const bool resumeDataset = false) {
  const std::string backendNode = kBackendNode;
  auto parametersClient =
      std::make_shared<rclcpp::AsyncParametersClient>(node, backendNode);

  if (!parametersClient->wait_for_service(1s)) {
    appendLog(ui, "Parameter service is not available for " +
                      QString::fromStdString(backendNode));
    return false;
  }

  auto future = parametersClient->set_parameters(
      collectParameters(ui, overwriteDataset, resumeDataset));
  if (future.wait_for(2s) != std::future_status::ready) {
    appendLog(ui, "Timed out while setting backend parameters");
    return false;
  }

  appendLog(ui, "Backend parameters updated");
  return true;
}

void updateStateFromJson(Ui::HandEyeFrontend &ui, const QString &json) {
  const QJsonDocument document = QJsonDocument::fromJson(json.toUtf8());
  if (!document.isObject()) {
    return;
  }

  const QJsonObject state = document.object();
  ui.progressBar->setValue(
      static_cast<int>(std::round(state["progress_percent"].toDouble(0.0))));
  const bool datasetReadyForCalibration =
      state["progress_percent"].toDouble(0.0) >= 100.0;
  ui.calibrateButton->setEnabled(datasetReadyForCalibration);
  ui.calibrateButton->setToolTip(
      datasetReadyForCalibration
          ? "Run hand-eye calibration."
          : "Calibration is enabled when dataset quality reaches 100%.");
  const int imageCount =
      state["image_count"].isNull() ? 0 : state["image_count"].toInt();

  ui.statsList->clear();
  ui.statsList->addItem(QString("Samples: %1").arg(imageCount));

  const QJsonObject stats = state["statistics"].toObject();
  if (!stats.isEmpty()) {
    if (ui.translationStatsCheck->isChecked()) {
      ui.statsList->addItem(
          QString("Translation coverage: %1%")
              .arg(stats["translation_coverage_score"].toDouble() * 100.0, 0,
                   'f', 0));
      ui.statsList->addItem(
          QString("  span x/y/z [m]: %1 / %2 / %3")
              .arg(stats["translation_spread_x"].toDouble(), 0, 'f', 5)
              .arg(stats["translation_spread_y"].toDouble(), 0, 'f', 5)
              .arg(stats["translation_spread_z"].toDouble(), 0, 'f', 5));
    }
    if (ui.rotationStatsCheck->isChecked()) {
      ui.statsList->addItem(
          QString("Rotation coverage: %1%")
              .arg(stats["rotation_coverage_score"].toDouble() * 100.0, 0, 'f',
                   0));
      ui.statsList->addItem(
          QString("  spread [deg]: %1")
              .arg(stats["rotation_spread_deg"].toDouble(), 0, 'f', 2));
    }
  } else {
    if (ui.translationStatsCheck->isChecked()) {
      ui.statsList->addItem("Translation coverage: waiting for poses");
    }
    if (ui.rotationStatsCheck->isChecked()) {
      ui.statsList->addItem("Rotation coverage: waiting for poses");
    }
  }
  if (ui.validationRmsStatsCheck->isChecked() &&
      !state["last_validation_rms"].isNull()) {
    ui.statsList->addItem(
        QString("Target lock RMS [m]: %1")
            .arg(state["last_validation_rms"].toDouble(), 0, 'f', 6));
  } else if (ui.validationRmsStatsCheck->isChecked()) {
    const QString status = state["last_validation_rms_status"].toString();
    ui.statsList->addItem("Target lock RMS [m]: " +
                          (status.isEmpty() ? "waiting for samples" : status));
  }
}

void showDonationDialog(QWidget &parent) {
  QDialog dialog(&parent);
  dialog.setWindowTitle("Buy me a coffee");
  auto *layout = new QVBoxLayout(&dialog);

  auto *imageLabel = new QLabel(&dialog);
  imageLabel->setAlignment(Qt::AlignCenter);
  const QPixmap donationImage(":/hand_eye_calibration/donation_usdt_erc20.png");
  imageLabel->setPixmap(donationImage.scaled(520, 640, Qt::KeepAspectRatio,
                                             Qt::SmoothTransformation));
  layout->addWidget(imageLabel);

  auto *closeButton = new QPushButton("Close", &dialog);
  QObject::connect(closeButton, &QPushButton::clicked, &dialog,
                   &QDialog::accept);
  layout->addWidget(closeButton, 0, Qt::AlignRight);
  dialog.exec();
}

void updatePatternStack(Ui::HandEyeFrontend &ui, const int index) {
  ui.patternStack->setCurrentIndex(index);
  ui.patternStack->setVisible(index > 0);
  if (index <= 0) {
    ui.patternStack->setFixedHeight(0);
    ui.patternStack->updateGeometry();
    return;
  }

  const QWidget *page = ui.patternStack->currentWidget();
  const int pageHeight = page ? page->sizeHint().height() : 0;
  ui.patternStack->setFixedHeight(std::max(1, pageHeight + 8));
  ui.patternStack->updateGeometry();
}

void configureVisuals(Ui::HandEyeFrontend &ui) {
  QFont titleFont = ui.settingsTitleLabel->font();
  titleFont.setPointSize(titleFont.pointSize() + 8);
  titleFont.setBold(true);
  ui.settingsTitleLabel->setFont(titleFont);

  ui.creditsLabel->setTextFormat(Qt::RichText);
  ui.creditsLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);
  ui.creditsLabel->setStyleSheet("color: #666; font-size: 11px;");
  const auto groupStyle = [](const QString &color) {
    return QString("QGroupBox {"
                   "  border: 1px solid #d8dde6;"
                   "  border-radius: 6px;"
                   "  margin-top: 14px;"
                   "  padding-top: 10px;"
                   "}"
                   "QGroupBox::title {"
                   "  subcontrol-origin: margin;"
                   "  left: 12px;"
                   "  padding: 1px 8px;"
                   "  color: %1;"
                   "  font-weight: 700;"
                   "  background: palette(window);"
                   "}")
        .arg(color);
  };
  ui.requiredGroup->setStyleSheet(groupStyle("#2f6fdd"));
  ui.optionalGroup->setStyleSheet(groupStyle("#5b6470"));
  ui.statisticsOptionsGroup->setStyleSheet(groupStyle("#147a5d"));
  ui.statsGroup->setStyleSheet(groupStyle("#147a5d"));
  ui.cameraGroup->setStyleSheet(groupStyle("#2f6fdd"));
  ui.cameraPreviewLabel->setStyleSheet(
      "background: #111820; color: #d9e1ea; border-radius: 6px;");

  updatePatternStack(ui, 0);
  ui.logGroup->setVisible(true);

  ui.armTopicLabel->setToolTip(
      "Arm pose/joint topic used to save gripper samples.");
  ui.imageTopicLabel->setToolTip(
      "Camera image topic used to capture calibration images.");
  ui.datasetPathLabel->setToolTip(
      "Output folder where captured calibration images and poses.csv are "
      "written.");
  ui.patternTypeLabel->setToolTip("Calibration target type.");
  ui.posesFormatLabel->setToolTip("Encoding used by each row in poses.csv:\n"
                                  "1: x y z qx qy qz qw\n"
                                  "2: x y z qw qx qy qz\n"
                                  "3: x y z roll pitch yaw [rad]\n"
                                  "4: x y z roll pitch yaw [deg]\n"
                                  "5: x y z yaw pitch roll [rad]\n"
                                  "6: x y z yaw pitch roll [deg]\n"
                                  "7: joint positions j0 j1 ... jn");
  ui.calibrationPathLabel->setToolTip(
      "Output YAML file where the computed calibration is written.");
  ui.intrinsicsPathLabel->setToolTip(
      "YAML with camera K and D. If empty, intrinsics are estimated "
      "from the dataset.");
  ui.eyeToHandCheck->setToolTip(
      "Use eye-to-hand mode. Disable for eye-in-hand calibration.");
  ui.translationStatsCheck->setToolTip(
      "Shows how well saved gripper positions cover translation space.");
  ui.rotationStatsCheck->setToolTip(
      "Shows how well saved gripper poses cover rotation space.");
  ui.validationRmsStatsCheck->setToolTip(
      "How much the target position moves in the gripper frame. Closer to 0 "
      "is better.");
  ui.arucoDictLabel->setToolTip("ArUco dictionary: 4x4, 5x5, 6x6, or 7x7.");
  ui.arucoIdLabel->setToolTip(
      "Marker id inside the selected ArUco dictionary.");
  ui.arucoSizeLabel->setToolTip("Physical marker size in meters.");
  ui.chessRowsLabel->setToolTip("Number of inner chessboard corner rows.");
  ui.chessColsLabel->setToolTip("Number of inner chessboard corner columns.");
  ui.chessCellSizeLabel->setToolTip("Physical chessboard cell size in meters.");
  ui.charucoRowsLabel->setToolTip("Number of ChArUco board rows.");
  ui.charucoColsLabel->setToolTip("Number of ChArUco board columns.");
  ui.charucoCellSizeLabel->setToolTip("Physical ChArUco cell size in meters.");
  ui.charucoMarkerSizeLabel->setToolTip(
      "Physical ChArUco marker size in meters.");
  ui.charucoDictLabel->setToolTip(
      "ArUco dictionary used by the ChArUco board.");
  ui.charucoIdLabel->setToolTip(
      "Marker id validation parameter for the ChArUco board.");
}

bool validateRequiredSettings(QWidget &window, Ui::HandEyeFrontend &ui) {
  QStringList missing;
  if (selectedTopic(ui.armTopicCombo).isEmpty()) {
    missing << "arm topic";
  }
  if (selectedTopic(ui.imageTopicCombo).isEmpty()) {
    missing << "camera topic";
  }
  if (ui.datasetPathEdit->text().trimmed().isEmpty()) {
    missing << "dataset output folder";
  }
  if (calibrationPath(ui).isEmpty()) {
    missing << "calibration output path";
  }
  if (ui.patternTypeCombo->currentIndex() == 0) {
    missing << "pattern type";
  }

  if (!missing.isEmpty()) {
    QMessageBox::warning(&window, "Missing settings",
                         "Fill required fields: " + missing.join(", "));
    return false;
  }
  return true;
}

struct DatasetOpenMode {
  bool overwrite = false;
  bool resume = false;
};

std::optional<DatasetOpenMode> askDatasetOpenMode(QWidget &window,
                                                  Ui::HandEyeFrontend &ui) {
  const QString datasetPath = ui.datasetPathEdit->text().trimmed();
  if (datasetPath.isEmpty()) {
    return DatasetOpenMode{};
  }

  QDir datasetDir(datasetPath);
  if (!datasetDir.exists()) {
    return DatasetOpenMode{};
  }

  const QFileInfoList entries = datasetDir.entryInfoList(
      QDir::NoDotAndDotDot | QDir::AllEntries | QDir::Hidden);
  if (entries.isEmpty()) {
    return DatasetOpenMode{};
  }

  QMessageBox messageBox(&window);
  messageBox.setWindowTitle("Dataset folder is not empty");
  messageBox.setText(
      "The selected dataset folder already exists and is not empty.");
  messageBox.setInformativeText(
      "Continue the existing dataset, start over by removing it, or cancel?");
  QPushButton *continueButton =
      messageBox.addButton("Continue Dataset", QMessageBox::AcceptRole);
  QPushButton *startOverButton =
      messageBox.addButton("Start Over", QMessageBox::DestructiveRole);
  messageBox.addButton(QMessageBox::Cancel);
  messageBox.setDefaultButton(continueButton);
  messageBox.exec();

  if (messageBox.clickedButton() == continueButton) {
    return DatasetOpenMode{false, true};
  }
  if (messageBox.clickedButton() == startOverButton) {
    return DatasetOpenMode{true, false};
  }
  return std::nullopt;
}

std::optional<DatasetOpenMode>
validateDatasetPathMode(QWidget &window, Ui::HandEyeFrontend &ui) {
  const QString datasetPath = ui.datasetPathEdit->text().trimmed();
  QFileInfo datasetInfo(datasetPath);
  if (datasetInfo.exists() && !datasetInfo.isDir()) {
    const QMessageBox::StandardButton answer = QMessageBox::warning(
        &window, "Dataset path is not a folder",
        "The selected dataset path exists, but it is not a folder.\n\n"
        "Replace it with a new dataset folder?",
        QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    if (answer == QMessageBox::Yes) {
      return DatasetOpenMode{true, false};
    }
    return std::nullopt;
  }
  return askDatasetOpenMode(window, ui);
}

void callTrigger(const rclcpp::Node::SharedPtr &node, Ui::HandEyeFrontend &ui,
                 const std::string &service, bool refreshStateAfter = true,
                 std::function<void()> onSuccess = {});

void handleTriggerResponse(const rclcpp::Node::SharedPtr &node,
                           Ui::HandEyeFrontend &ui, const std::string &service,
                           const Trigger::Response::SharedPtr &response,
                           bool refreshStateAfter,
                           const std::function<void()> &onSuccess) {
  const QString prefix = response->success ? "[ok] " : "[failed] ";
  const QString message = (service == "get_state" && response->success)
                              ? QString("State refreshed")
                              : userMessage(response->message);
  appendLog(ui, prefix + QString::fromStdString(service) + ": " + message);
  if (service == "get_state") {
    updateStateFromJson(ui, QString::fromStdString(response->message));
  } else if (response->success && onSuccess) {
    onSuccess();
  } else if (refreshStateAfter) {
    callTrigger(node, ui, "get_state", false);
  }
}

void callTrigger(const rclcpp::Node::SharedPtr &node, Ui::HandEyeFrontend &ui,
                 const std::string &service, const bool refreshStateAfter,
                 std::function<void()> onSuccess) {
  const std::string backendNode = kBackendNode;
  const std::string fullServiceName = serviceName(backendNode, service);
  auto client = node->create_client<Trigger>(fullServiceName);

  if (!client->wait_for_service(500ms)) {
    appendLog(ui, "Service is not available: " +
                      QString::fromStdString(fullServiceName));
    return;
  }

  appendLog(ui, "Calling " + QString::fromStdString(fullServiceName));
  auto request = std::make_shared<Trigger::Request>();
  client->async_send_request(
      request, [node, &ui, service, client, refreshStateAfter,
                onSuccess](rclcpp::Client<Trigger>::SharedFuture future) {
        (void)client;
        const auto response = future.get();
        QMetaObject::invokeMethod(
            ui.logText,
            [node, &ui, service, response, refreshStateAfter, onSuccess]() {
              handleTriggerResponse(node, ui, service, response,
                                    refreshStateAfter, onSuccess);
            },
            Qt::QueuedConnection);
      });
}

void pushAndCall(const rclcpp::Node::SharedPtr &node, Ui::HandEyeFrontend &ui,
                 const std::string &service) {
  if (!pushParameters(node, ui)) {
    return;
  }
  callTrigger(node, ui, service);
}

std::optional<std::string>
topicTypeFromGraph(const rclcpp::Node::SharedPtr &node,
                   const std::string &topicName) {
  const auto topics = node->get_topic_names_and_types();
  const auto it = std::ranges::find_if(topics, [&topicName](const auto &entry) {
    return entry.first == topicName;
  });
  if (it == topics.end() || it->second.empty()) {
    return std::nullopt;
  }
  return it->second.front();
}

class CameraFeed final {
public:
  CameraFeed(const rclcpp::Node::SharedPtr &node, const std::string &topic,
             std::string type, QLabel *label, const std::string &patternInfo)
      : node_(node), label_(label) {
    if (type.empty()) {
      type = topicTypeFromGraph(node_, topic).value_or(kImageType);
    }
    if (!patternInfo.empty()) {
      try {
        detector_ =
            pattern_detector::PatternDetector::fromPatternInfo(patternInfo);
      } catch (const std::exception &) {
        detector_.reset();
      }
    }

    worker_ = std::thread([this]() { processFrames(); });
    timer_ = new QTimer(label_);
    timer_->setInterval(33);
    QObject::connect(timer_, &QTimer::timeout, label_, [this]() {
      QImage image;
      {
        std::lock_guard<std::mutex> lock(renderMutex_);
        if (renderImage_.isNull()) {
          return;
        }
        image = renderImage_;
      }
      label_->setPixmap(QPixmap::fromImage(image).scaled(
          label_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
    });
    timer_->start();

    if (type == kCompressedImageType) {
      const std::string compressedTopic =
          topic.ends_with("/compressed") ? topic : topic + "/compressed";
      compressedSub_ =
          node_->create_subscription<sensor_msgs::msg::CompressedImage>(
              compressedTopic, rclcpp::SensorDataQoS(),
              [this](
                  const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
                if (!shouldAcceptFrame()) {
                  return;
                }
                const cv::Mat encoded(msg->data, true);
                pushFrame(cv::imdecode(encoded, cv::IMREAD_COLOR));
              });
    } else {
      imageSub_ = node_->create_subscription<sensor_msgs::msg::Image>(
          topic, rclcpp::SensorDataQoS(),
          [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
            try {
              if (!shouldAcceptFrame()) {
                return;
              }
              pushFrame(
                  cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8)
                      ->image);
            } catch (const std::exception &) {
            }
          });
    }
  }

  ~CameraFeed() {
    imageSub_.reset();
    compressedSub_.reset();
    stop_.store(true);
    frameCv_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
    if (timer_) {
      timer_->stop();
    }
  }

private:
  bool shouldAcceptFrame() {
    std::lock_guard<std::mutex> lock(throttleMutex_);
    const auto now = std::chrono::steady_clock::now();
    if (now < nextFrameTime_) {
      return false;
    }
    nextFrameTime_ = now + std::chrono::milliseconds(33);
    return true;
  }

  void pushFrame(const cv::Mat &image) {
    if (image.empty()) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(frameMutex_);
      latestFrame_ = image.clone();
      hasFrame_ = true;
    }
    frameCv_.notify_one();
  }

  void drawDetection(cv::Mat &image) const {
    if (!detector_) {
      return;
    }

    std::vector<cv::Point2f> points;
    if (!detector_->detectImagePoints(image, points)) {
      return;
    }

    if (detector_->kind == pattern_detector::PatternKind::CHESSBOARD) {
      cv::drawChessboardCorners(image, detector_->getChessboardDims(), points,
                                true);
      return;
    }

    if (points.size() >= 2) {
      for (size_t i = 0; i < points.size(); ++i) {
        cv::line(image, points[i], points[(i + 1) % points.size()],
                 cv::Scalar(70, 220, 70), 2);
      }
    }
    for (const cv::Point2f &point : points) {
      cv::circle(image, point, 4, cv::Scalar(50, 190, 255), -1);
    }
  }

  void processFrames() {
    while (!stop_.load()) {
      cv::Mat image;
      {
        std::unique_lock<std::mutex> lock(frameMutex_);
        frameCv_.wait(lock, [this]() { return stop_.load() || hasFrame_; });
        if (stop_.load()) {
          return;
        }
        image = latestFrame_.clone();
        hasFrame_ = false;
      }

      drawDetection(image);

      cv::Mat rgb;
      if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
      } else {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
      }

      QImage qimage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                    QImage::Format_RGB888);
      {
        std::lock_guard<std::mutex> lock(renderMutex_);
        renderImage_ = qimage.copy();
      }
    }
  }

  rclcpp::Node::SharedPtr node_;
  QLabel *label_ = nullptr;
  QTimer *timer_ = nullptr;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
      compressedSub_;
  std::optional<pattern_detector::PatternDetector> detector_;
  std::thread worker_;
  std::atomic_bool stop_{false};
  std::mutex throttleMutex_;
  std::chrono::steady_clock::time_point nextFrameTime_{};
  std::condition_variable frameCv_;
  std::mutex frameMutex_;
  cv::Mat latestFrame_;
  bool hasFrame_ = false;
  std::mutex renderMutex_;
  QImage renderImage_;
};

class CameraWindow final : public QWidget {
public:
  CameraWindow(const rclcpp::Node::SharedPtr &node, const std::string &topic,
               std::string type, const std::string &patternInfo) {
    setWindowTitle("Camera View");
    resize(800, 600);
    auto *label = new QLabel(this);
    label->setAlignment(Qt::AlignCenter);
    label->setMinimumSize(640, 480);
    auto *layout = new QVBoxLayout(this);
    layout->addWidget(label);
    feed_ = std::make_unique<CameraFeed>(node, topic, std::move(type), label,
                                         patternInfo);
  }

private:
  std::unique_ptr<CameraFeed> feed_;
};

} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  QApplication app(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("hand_eye_frontend");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinThread([&executor]() { executor.spin(); });

  QMainWindow window;
  Ui::HandEyeFrontend ui;
  ui.setupUi(&window);
  ui.mainStack->setCurrentWidget(ui.settingsPage);
  ui.patternStack->setCurrentIndex(0);
  ui.creditsLabel->setOpenExternalLinks(false);
  ui.mainSplitter->setChildrenCollapsible(true);
  ui.mainSplitter->setCollapsible(1, true);
  ui.mainSplitter->setStretchFactor(0, 1);
  ui.mainSplitter->setStretchFactor(1, 0);
  ui.mainSplitter->setSizes({560, 180});
  configureVisuals(ui);
  ui.workflowContentSplitter->setChildrenCollapsible(false);
  ui.workflowContentSplitter->setStretchFactor(0, 0);
  ui.workflowContentSplitter->setStretchFactor(1, 1);
  ui.workflowContentSplitter->setSizes({320, 640});
  ui.calibrateButton->setEnabled(false);

  std::unique_ptr<CameraFeed> cameraPreview;
  auto startEmbeddedCameraPreview = [&]() {
    const std::string topic = selectedTopic(ui.imageTopicCombo).toStdString();
    if (topic.empty()) {
      ui.cameraPreviewLabel->setText("Choose a camera topic in Settings.");
      return;
    }
    ui.cameraPreviewLabel->setText("Waiting for camera frames...");
    cameraPreview = std::make_unique<CameraFeed>(
        node, topic, selectedTopicType(ui.imageTopicCombo).toStdString(),
        ui.cameraPreviewLabel, patternInfo(ui).toStdString());
  };

  QObject::connect(ui.patternTypeCombo,
                   QOverload<int>::of(&QComboBox::currentIndexChanged),
                   [&](const int index) { updatePatternStack(ui, index); });
  QObject::connect(ui.refreshTopicsButton, &QPushButton::clicked,
                   [&]() { refreshGraph(node, ui); });
  QObject::connect(ui.creditsLabel, &QLabel::linkActivated,
                   [&](const QString &link) {
                     if (link == "donate://coffee") {
                       showDonationDialog(window);
                       return;
                     }
                     QDesktopServices::openUrl(QUrl(link));
                   });
  QObject::connect(ui.datasetBrowseButton, &QPushButton::clicked, [&]() {
    const QString path =
        QFileDialog::getExistingDirectory(&window, "Select dataset folder");
    if (!path.isEmpty()) {
      ui.datasetPathEdit->setText(path);
    }
  });
  QObject::connect(ui.calibrationBrowseButton, &QPushButton::clicked, [&]() {
    const QString initialPath = calibrationPath(ui);
    const QString path =
        QFileDialog::getSaveFileName(&window, "Select calibration YAML",
                                     initialPath, "YAML files (*.yaml *.yml)");
    if (!path.isEmpty()) {
      ui.calibrationPathEdit->setText(path);
    }
  });
  QObject::connect(ui.intrinsicsBrowseButton, &QPushButton::clicked, [&]() {
    const QString path = QFileDialog::getOpenFileName(
        &window, "Select intrinsics YAML", {}, "YAML files (*.yaml *.yml)");
    if (!path.isEmpty()) {
      ui.intrinsicsPathEdit->setText(path);
    }
  });
  QObject::connect(ui.saveSettingsButton, &QPushButton::clicked, [&]() {
    saveSettings(ui);
    appendLog(ui, "Settings saved");
  });
  QObject::connect(ui.nextButton, &QPushButton::clicked, [&]() {
    if (!validateRequiredSettings(window, ui)) {
      return;
    }
    const std::optional<DatasetOpenMode> datasetMode =
        validateDatasetPathMode(window, ui);
    if (!datasetMode) {
      appendLog(ui, "Dataset selection canceled");
      return;
    }
    saveSettings(ui);
    if (!pushParameters(node, ui, datasetMode->overwrite,
                        datasetMode->resume)) {
      return;
    }
    ui.mainStack->setCurrentWidget(ui.workflowPage);
    startEmbeddedCameraPreview();
    callTrigger(node, ui, "configure", true, [&]() {
      callTrigger(node, ui, "prepare_dataset", true,
                  [&]() { callTrigger(node, ui, "start_collection"); });
    });
  });
  QObject::connect(ui.backToSettingsButton, &QPushButton::clicked, [&]() {
    cameraPreview.reset();
    ui.cameraPreviewLabel->setText("Camera stream will appear after setup.");
    ui.mainStack->setCurrentWidget(ui.settingsPage);
  });
  QObject::connect(ui.captureSampleButton, &QPushButton::clicked, [&]() {
    if (!pushParameters(node, ui)) {
      return;
    }
    callTrigger(node, ui, "capture_sample");
  });
  QObject::connect(ui.removeSampleButton, &QPushButton::clicked,
                   [&]() { callTrigger(node, ui, "remove_sample"); });
  QObject::connect(ui.calibrateButton, &QPushButton::clicked, [&]() {
    if (!ui.calibrateButton->isEnabled()) {
      appendLog(ui, "Dataset quality is not ready for calibration yet");
      return;
    }
    pushAndCall(node, ui, "run_calibration");
  });

  refreshGraph(node, ui);
  loadSettings(ui);
  updatePatternStack(ui, ui.patternTypeCombo->currentIndex());

  appendLog(ui, "Frontend started");
  window.show();

  const int result = app.exec();
  executor.cancel();
  if (spinThread.joinable()) {
    spinThread.join();
  }
  rclcpp::shutdown();
  return result;
}
