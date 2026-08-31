#include "app/frontend_state.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>

#include <cmath>

namespace hand_eye::app {
namespace {

std::optional<int> optionalInteger(const QJsonObject &object,
                                   const char *name) {
  const QJsonValue value = object.value(name);
  if (!value.isDouble()) {
    return std::nullopt;
  }
  return value.toInt();
}

std::optional<double> optionalDouble(const QJsonObject &object,
                                     const char *name) {
  const QJsonValue value = object.value(name);
  if (!value.isDouble()) {
    return std::nullopt;
  }
  return value.toDouble();
}

QString stringOrDefault(const QJsonObject &object, const char *name,
                        const QString &defaultValue) {
  const QJsonValue value = object.value(name);
  if (!value.isString()) {
    return defaultValue;
  }
  const QString parsed = value.toString().trimmed();
  return parsed.isEmpty() ? defaultValue : parsed;
}

} // namespace

QString FrontendState::serializePattern(const PatternSettings &pattern) {
  switch (pattern.type) {
  case 1:
    return QString("1 %1 %2 %3")
        .arg(pattern.arucoDictionary)
        .arg(pattern.arucoId)
        .arg(pattern.arucoMarkerSize, 0, 'f', 4);
  case 2:
    return QString("2 %1 %2 %3")
        .arg(pattern.chessRows)
        .arg(pattern.chessColumns)
        .arg(pattern.chessCellSize, 0, 'f', 4);
  case 3:
    return QString("3 %1 %2 %3 %4 %5 %6")
        .arg(pattern.charucoRows)
        .arg(pattern.charucoColumns)
        .arg(pattern.charucoCellSize, 0, 'f', 4)
        .arg(pattern.charucoMarkerSize, 0, 'f', 4)
        .arg(pattern.charucoDictionary)
        .arg(pattern.charucoId);
  default:
    return {};
  }
}

std::optional<PatternSettings>
FrontendState::parsePattern(const QString &value) {
  const QStringList fields = value.simplified().split(' ', Qt::SkipEmptyParts);
  if (fields.isEmpty()) {
    return std::nullopt;
  }

  bool valid = false;
  PatternSettings pattern;
  pattern.type = fields[0].toInt(&valid);
  if (!valid) {
    return std::nullopt;
  }
  auto integer = [&fields](const int index, bool &ok) {
    if (index >= fields.size()) {
      ok = false;
      return 0;
    }
    return fields[index].toInt(&ok);
  };
  auto number = [&fields](const int index, bool &ok) {
    if (index >= fields.size()) {
      ok = false;
      return 0.0;
    }
    return fields[index].toDouble(&ok);
  };

  switch (pattern.type) {
  case 1:
    if (fields.size() != 4) {
      return std::nullopt;
    }
    pattern.arucoDictionary = integer(1, valid);
    if (!valid)
      return std::nullopt;
    pattern.arucoId = integer(2, valid);
    if (!valid)
      return std::nullopt;
    pattern.arucoMarkerSize = number(3, valid);
    break;
  case 2:
    if (fields.size() != 4) {
      return std::nullopt;
    }
    pattern.chessRows = integer(1, valid);
    if (!valid)
      return std::nullopt;
    pattern.chessColumns = integer(2, valid);
    if (!valid)
      return std::nullopt;
    pattern.chessCellSize = number(3, valid);
    break;
  case 3:
    if (fields.size() != 7) {
      return std::nullopt;
    }
    pattern.charucoRows = integer(1, valid);
    if (!valid)
      return std::nullopt;
    pattern.charucoColumns = integer(2, valid);
    if (!valid)
      return std::nullopt;
    pattern.charucoCellSize = number(3, valid);
    if (!valid)
      return std::nullopt;
    pattern.charucoMarkerSize = number(4, valid);
    if (!valid)
      return std::nullopt;
    pattern.charucoDictionary = integer(5, valid);
    if (!valid)
      return std::nullopt;
    pattern.charucoId = integer(6, valid);
    break;
  default:
    return std::nullopt;
  }
  return valid ? std::optional<PatternSettings>(pattern) : std::nullopt;
}

std::optional<BackendState>
FrontendState::parseBackendState(const QString &json) {
  QJsonParseError error;
  const QJsonDocument document = QJsonDocument::fromJson(json.toUtf8(), &error);
  if (error.error != QJsonParseError::NoError || !document.isObject()) {
    return std::nullopt;
  }

  const QJsonObject object = document.object();
  BackendState state;
  state.configured = object.value("configured").toBool();
  state.collecting = object.value("collecting").toBool();
  state.datasetPrepared = object.value("dataset_prepared").toBool();
  state.armTopic = object.value("arm_topic").toString();
  state.imageTopic = object.value("image_topic").toString();
  state.cameraInfoTopic = object.value("camera_info_topic").toString();
  state.datasetPath = object.value("dataset_path").toString();
  state.calibrationPath = object.value("calibration_path").toString();
  state.patternInfo = object.value("pattern_info").toString();
  state.intrinsicsPath = object.value("intrinsics_path").toString();
  state.posesFormat = object.value("poses_format").toInt(1);
  state.eyeToHand = object.value("eye_to_hand").toBool(true);
  state.robotBaseFrame = object.value("robot_base_frame").toString();
  state.robotEffectorFrame = object.value("robot_effector_frame").toString();
  state.cameraFrame = object.value("camera_frame").toString();
  state.targetFrame = object.value("target_frame").toString();
  state.armTopicType = object.value("arm_topic_type").toString();
  state.imageTopicType = object.value("image_topic_type").toString();
  state.minimumSamples = object.value("min_samples").toInt();
  state.nextSampleIndex = object.value("next_sample_index").toInt();
  state.progressPercent = object.value("progress_percent").toDouble();
  state.imageCount = optionalInteger(object, "image_count");
  state.poseCount = optionalInteger(object, "pose_count");
  state.lastValidationRms = optionalDouble(object, "last_validation_rms");
  state.lastValidationRmsStatus =
      object.value("last_validation_rms_status").toString();
  state.lastError = object.value("last_error").toString();
  state.targetTfVisible = object.value("target_tf_visible").toBool();
  state.calibrationTfAvailable =
      object.value("calibration_tf_available").toBool();
  state.targetReprojectionPixels =
      optionalDouble(object, "target_reprojection_px");
  if (const auto timeConstant =
          optionalDouble(object, "target_pose_filter_time_constant_s");
      timeConstant && std::isfinite(*timeConstant) && *timeConstant >= 0.0) {
    state.targetPoseFilterTimeConstantSeconds = *timeConstant;
  }
  const QJsonValue filterEnabled = object.value("target_pose_filter_enabled");
  state.targetPoseFilterEnabled =
      filterEnabled.isBool() ? filterEnabled.toBool()
                             : state.targetPoseFilterTimeConstantSeconds > 0.0;
  const QJsonValue filterInitialized =
      object.value("target_pose_filter_initialized");
  state.targetPoseFilterInitialized =
      filterInitialized.isBool() && filterInitialized.toBool();

  const QJsonValue cameraSourceValue = object.value("camera_tf_source");
  if (cameraSourceValue.isString()) {
    const QString source = cameraSourceValue.toString().trimmed().toLower();
    if (source == "initial" || source == "online" || source == "none") {
      state.cameraTfSource = source;
    }
  } else if (state.calibrationTfAvailable) {
    // Backends predating camera_tf_source only published a transform after a
    // calibration result was available.
    state.cameraTfSource = "online";
  }

  const QJsonValue cameraActiveValue = object.value("camera_tf_active");
  state.cameraTfActive = cameraActiveValue.isBool()
                             ? cameraActiveValue.toBool()
                             : state.cameraTfSource != "none";
  state.cameraTfParentFrame =
      stringOrDefault(object, "camera_tf_parent_frame", {});
  if (state.cameraTfParentFrame.isEmpty() && state.cameraTfActive) {
    state.cameraTfParentFrame =
        state.eyeToHand ? state.robotBaseFrame : state.robotEffectorFrame;
  }

  const QJsonValue initialConfiguredValue =
      object.value("initial_camera_tf_configured");
  state.initialCameraTfConfigured = initialConfiguredValue.isBool()
                                        ? initialConfiguredValue.toBool()
                                        : state.cameraTfSource == "initial";

  const QJsonValue gripperActiveValue = object.value("gripper_tf_active");
  state.gripperTfActive =
      gripperActiveValue.isBool() && gripperActiveValue.toBool();
  state.gripperVisualizationFrame = stringOrDefault(
      object, "gripper_visualization_frame", state.gripperVisualizationFrame);
  state.targetMarkersTopic =
      stringOrDefault(object, "target_markers_topic", state.targetMarkersTopic);

  const QJsonObject statistics = object.value("statistics").toObject();
  if (!statistics.isEmpty()) {
    BackendStatistics parsed;
    parsed.samples = statistics.value("samples").toInt();
    parsed.validPoseRows = statistics.value("valid_pose_rows").toInt();
    parsed.translationSpreadX =
        statistics.value("translation_spread_x").toDouble();
    parsed.translationSpreadY =
        statistics.value("translation_spread_y").toDouble();
    parsed.translationSpreadZ =
        statistics.value("translation_spread_z").toDouble();
    parsed.rotationSpreadDegrees =
        statistics.value("rotation_spread_deg").toDouble();
    parsed.translationCoverageScore =
        statistics.value("translation_coverage_score").toDouble();
    parsed.rotationCoverageScore =
        statistics.value("rotation_coverage_score").toDouble();
    state.statistics = parsed;
  }

  state.calibrationMethod = object.value("calibration_method").toString();
  state.translationRmsMeters = optionalDouble(object, "translation_rms_m");
  state.rotationRmsDegrees = optionalDouble(object, "rotation_rms_deg");
  state.rejectedSampleCount = optionalInteger(object, "rejected_sample_count");
  const QJsonArray rejected = object.value("rejected_sample_indices").toArray();
  state.rejectedSampleIndices.reserve(rejected.size());
  for (const QJsonValue &value : rejected) {
    if (value.isDouble() && value.toInt(-1) >= 0) {
      state.rejectedSampleIndices.push_back(value.toInt());
    }
  }
  return state;
}

FrontendViewState
FrontendState::makeViewState(const BackendState &state,
                             const FrontendSettings &settings) {
  FrontendViewState view;
  view.progressValue = static_cast<int>(std::round(state.progressPercent));
  view.calibrationReady = state.progressPercent >= 100.0;
  view.statisticsLines
      << QString("Samples: %1").arg(state.imageCount.value_or(0));

  if (state.statistics) {
    if (settings.translationStatistics) {
      view.statisticsLines
          << QString("Translation coverage: %1%")
                 .arg(state.statistics->translationCoverageScore * 100.0, 0,
                      'f', 0)
          << QString("  span x/y/z [m]: %1 / %2 / %3")
                 .arg(state.statistics->translationSpreadX, 0, 'f', 5)
                 .arg(state.statistics->translationSpreadY, 0, 'f', 5)
                 .arg(state.statistics->translationSpreadZ, 0, 'f', 5);
    }
    if (settings.rotationStatistics) {
      view.statisticsLines << QString("Rotation coverage: %1%")
                                  .arg(state.statistics->rotationCoverageScore *
                                           100.0,
                                       0, 'f', 0)
                           << QString("  spread [deg]: %1")
                                  .arg(state.statistics->rotationSpreadDegrees,
                                       0, 'f', 2);
    }
  } else {
    if (settings.translationStatistics) {
      view.statisticsLines << "Translation coverage: waiting for poses";
    }
    if (settings.rotationStatistics) {
      view.statisticsLines << "Rotation coverage: waiting for poses";
    }
  }
  if (settings.validationRmsStatistics) {
    view.statisticsLines << (state.lastValidationRms
                                 ? QString("Target lock RMS [m]: %1")
                                       .arg(*state.lastValidationRms, 0, 'f', 6)
                                 : "Target lock RMS [m]: " +
                                       (state.lastValidationRmsStatus.isEmpty()
                                            ? QString("waiting for samples")
                                            : state.lastValidationRmsStatus));
  }
  if (!state.calibrationMethod.isEmpty()) {
    view.statisticsLines << "Calibration method: " + state.calibrationMethod;
  }
  if (state.translationRmsMeters && state.rotationRmsDegrees) {
    view.statisticsLines << QString("Calibration RMS: %1 m / %2 deg")
                                .arg(*state.translationRmsMeters, 0, 'f', 6)
                                .arg(*state.rotationRmsDegrees, 0, 'f', 3);
  }
  if (state.rejectedSampleCount) {
    view.statisticsLines
        << QString("Rejected samples: %1").arg(*state.rejectedSampleCount);
  }
  if (!state.rejectedSampleIndices.empty()) {
    QStringList indices;
    for (const int index : state.rejectedSampleIndices) {
      indices << QString::number(index);
    }
    view.statisticsLines << "Rejected sample indices: " + indices.join(", ");
  }
  view.statisticsLines
      << (state.targetTfVisible
              ? QString("Live target TF: visible (%1 px reprojection)")
                    .arg(state.targetReprojectionPixels.value_or(0.0), 0, 'f',
                         3)
              : QString("Live target TF: waiting for target"));
  if (state.targetPoseFilterEnabled) {
    view.statisticsLines << QString("Marker smoothing: %1 s (%2)")
                                .arg(state.targetPoseFilterTimeConstantSeconds,
                                     0, 'f', 2)
                                .arg(state.targetPoseFilterInitialized
                                         ? "active"
                                         : "waiting for target");
  } else {
    view.statisticsLines << "Marker smoothing: off (raw pose)";
  }
  if (state.cameraTfActive && state.cameraTfSource == "online") {
    view.statisticsLines << "Live camera TF: online estimate active";
  } else if (state.cameraTfActive && state.cameraTfSource == "initial") {
    QString transform;
    if (!state.cameraTfParentFrame.isEmpty() && !state.cameraFrame.isEmpty()) {
      transform = QString(" (%1 -> %2)")
                      .arg(state.cameraTfParentFrame, state.cameraFrame);
    }
    view.statisticsLines << "Live camera TF: initial guess active" + transform;
  } else if (state.cameraTfActive) {
    view.statisticsLines << "Live camera TF: active";
  } else if (state.calibrationTfAvailable) {
    view.statisticsLines
        << "Live camera TF: online estimate available, publication inactive";
  } else if (state.initialCameraTfConfigured) {
    view.statisticsLines
        << "Live camera TF: initial guess configured, publication inactive";
  } else {
    view.statisticsLines << "Live camera TF: waiting for solvable dataset";
  }

  view.statisticsLines
      << (state.gripperTfActive
              ? QString("Live gripper TF: visible (%1)")
                    .arg(state.gripperVisualizationFrame)
              : QString("Live gripper TF: waiting for gripper pose"));
  return view;
}

QString FrontendState::humanizeBackendMessage(const QString &message) {
  QString text = message.trimmed();
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

void FrontendState::save(QSettings &settings, const FrontendSettings &value) {
  settings.setValue("saved", true);
  settings.setValue("arm_topic", value.armTopic);
  settings.setValue("image_topic", value.imageTopic);
  settings.setValue("dataset_path", value.datasetPath);
  settings.setValue("calibration_path", value.calibrationPath);
  settings.setValue("intrinsics_path", value.intrinsicsPath);
  settings.setValue("pattern_type", value.pattern.type);
  settings.setValue("poses_format", value.posesFormat);
  settings.setValue("eye_to_hand", value.eyeToHand);
  settings.setValue("translation_stats", value.translationStatistics);
  settings.setValue("rotation_stats", value.rotationStatistics);
  settings.setValue("validation_rms_stats", value.validationRmsStatistics);
  settings.setValue("aruco_dict", value.pattern.arucoDictionary);
  settings.setValue("aruco_id", value.pattern.arucoId);
  settings.setValue("aruco_marker_size", value.pattern.arucoMarkerSize);
  settings.setValue("chess_rows", value.pattern.chessRows);
  settings.setValue("chess_cols", value.pattern.chessColumns);
  settings.setValue("chess_cell_size", value.pattern.chessCellSize);
  settings.setValue("charuco_rows", value.pattern.charucoRows);
  settings.setValue("charuco_cols", value.pattern.charucoColumns);
  settings.setValue("charuco_cell_size", value.pattern.charucoCellSize);
  settings.setValue("charuco_marker_size", value.pattern.charucoMarkerSize);
  settings.setValue("charuco_dict", value.pattern.charucoDictionary);
  settings.setValue("charuco_id", value.pattern.charucoId);
  settings.sync();
}

std::optional<FrontendSettings> FrontendState::load(const QSettings &settings) {
  if (!settings.value("saved", false).toBool()) {
    return std::nullopt;
  }

  FrontendSettings value;
  value.armTopic = settings.value("arm_topic").toString();
  value.imageTopic = settings.value("image_topic").toString();
  value.datasetPath = settings.value("dataset_path").toString();
  value.calibrationPath =
      settings.value("calibration_path", "calibration.yaml").toString();
  if (value.calibrationPath.trimmed().isEmpty()) {
    value.calibrationPath = "calibration.yaml";
  }
  value.intrinsicsPath = settings.value("intrinsics_path").toString();
  value.pattern.type = settings.value("pattern_type", 0).toInt();
  value.posesFormat = settings.value("poses_format", 1).toInt();
  value.eyeToHand = settings.value("eye_to_hand", false).toBool();
  value.translationStatistics =
      settings.value("translation_stats", true).toBool();
  value.rotationStatistics = settings.value("rotation_stats", true).toBool();
  value.validationRmsStatistics =
      settings.value("validation_rms_stats", false).toBool();
  value.pattern.arucoDictionary = settings.value("aruco_dict", 4).toInt();
  value.pattern.arucoId = settings.value("aruco_id", 0).toInt();
  value.pattern.arucoMarkerSize =
      settings.value("aruco_marker_size", 0.0).toDouble();
  value.pattern.chessRows = settings.value("chess_rows", 1).toInt();
  value.pattern.chessColumns = settings.value("chess_cols", 1).toInt();
  value.pattern.chessCellSize =
      settings.value("chess_cell_size", 0.0).toDouble();
  value.pattern.charucoRows = settings.value("charuco_rows", 1).toInt();
  value.pattern.charucoColumns = settings.value("charuco_cols", 1).toInt();
  value.pattern.charucoCellSize =
      settings.value("charuco_cell_size", 0.0).toDouble();
  value.pattern.charucoMarkerSize =
      settings.value("charuco_marker_size", 0.0).toDouble();
  value.pattern.charucoDictionary = settings.value("charuco_dict", 4).toInt();
  value.pattern.charucoId = settings.value("charuco_id", 0).toInt();
  return value;
}

} // namespace hand_eye::app
