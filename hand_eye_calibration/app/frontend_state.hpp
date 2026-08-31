#pragma once

#include <QSettings>
#include <QString>
#include <QStringList>

#include <optional>
#include <vector>

namespace hand_eye::app {

struct PatternSettings {
  int type = 0;

  int arucoDictionary = 4;
  int arucoId = 0;
  double arucoMarkerSize = 0.0;

  int chessRows = 1;
  int chessColumns = 1;
  double chessCellSize = 0.0;

  int charucoRows = 1;
  int charucoColumns = 1;
  double charucoCellSize = 0.0;
  double charucoMarkerSize = 0.0;
  int charucoDictionary = 4;
  int charucoId = 0;
};

struct FrontendSettings {
  QString armTopic;
  QString imageTopic;
  QString imageTopicType;
  QString datasetPath;
  QString calibrationPath = "calibration.yaml";
  QString intrinsicsPath;
  PatternSettings pattern;
  int posesFormat = 1;
  bool eyeToHand = false;
  bool translationStatistics = true;
  bool rotationStatistics = true;
  bool validationRmsStatistics = false;
};

struct BackendStatistics {
  int samples = 0;
  int validPoseRows = 0;
  double translationSpreadX = 0.0;
  double translationSpreadY = 0.0;
  double translationSpreadZ = 0.0;
  double rotationSpreadDegrees = 0.0;
  double translationCoverageScore = 0.0;
  double rotationCoverageScore = 0.0;
};

struct BackendState {
  bool configured = false;
  bool collecting = false;
  bool datasetPrepared = false;
  QString armTopic;
  QString imageTopic;
  QString cameraInfoTopic;
  QString datasetPath;
  QString calibrationPath;
  QString patternInfo;
  QString intrinsicsPath;
  int posesFormat = 1;
  bool eyeToHand = true;
  QString robotBaseFrame;
  QString robotEffectorFrame;
  QString cameraFrame;
  QString targetFrame;
  QString armTopicType;
  QString imageTopicType;
  int minimumSamples = 0;
  int nextSampleIndex = 0;
  double progressPercent = 0.0;
  std::optional<int> imageCount;
  std::optional<int> poseCount;
  std::optional<BackendStatistics> statistics;
  std::optional<double> lastValidationRms;
  QString lastValidationRmsStatus;
  QString lastError;
  bool targetTfVisible = false;
  bool calibrationTfAvailable = false;
  std::optional<double> targetReprojectionPixels;
  double targetPoseFilterTimeConstantSeconds = 0.0;
  bool targetPoseFilterEnabled = false;
  bool targetPoseFilterInitialized = false;
  QString cameraTfSource = "none";
  bool cameraTfActive = false;
  QString cameraTfParentFrame;
  bool initialCameraTfConfigured = false;
  bool gripperTfActive = false;
  QString gripperVisualizationFrame = "hand_eye_gripper_pose";
  QString targetMarkersTopic = "/hand_eye_backend/target_markers";

  // Optional fields emitted by the robust backend. Keeping them optional makes
  // the frontend compatible with older backend versions.
  QString calibrationMethod;
  std::optional<double> translationRmsMeters;
  std::optional<double> rotationRmsDegrees;
  std::optional<int> rejectedSampleCount;
  std::vector<int> rejectedSampleIndices;
};

struct FrontendViewState {
  int progressValue = 0;
  bool calibrationReady = false;
  QStringList statisticsLines;
};

class FrontendState final {
public:
  static QString serializePattern(const PatternSettings &pattern);
  static std::optional<PatternSettings> parsePattern(const QString &value);
  static std::optional<BackendState> parseBackendState(const QString &json);
  static FrontendViewState makeViewState(const BackendState &state,
                                         const FrontendSettings &settings);
  static QString humanizeBackendMessage(const QString &message);

  static void save(QSettings &settings, const FrontendSettings &value);
  static std::optional<FrontendSettings> load(const QSettings &settings);
};

} // namespace hand_eye::app
