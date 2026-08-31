#include "lib/core/target.hpp"

#include <gtest/gtest.h>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>

namespace hand_eye::core {
namespace {

TEST(TargetTest, ParsesAllSupportedPatternFormats) {
  const PatternConfig aruco = parsePattern("1 4 7 0.04");
  EXPECT_EQ(aruco.type, PatternType::ARUCO);
  EXPECT_EQ(aruco.markerId, 7);
  EXPECT_DOUBLE_EQ(aruco.markerSizeMeters, 0.04);

  const PatternConfig chessboard = parsePattern("2 6 9 0.025");
  EXPECT_EQ(chessboard.type, PatternType::CHESSBOARD);
  EXPECT_EQ(chessboard.rows, 6);
  EXPECT_EQ(chessboard.columns, 9);

  const PatternConfig charuco = parsePattern("3 8 6 0.04 0.03 4");
  EXPECT_EQ(charuco.type, PatternType::CHARUCO);
  EXPECT_EQ(charuco.rows, 8);
  EXPECT_EQ(charuco.columns, 6);

  EXPECT_THROW((void)parsePattern("1 8 0 0.04"), std::invalid_argument);
  EXPECT_THROW((void)parsePattern("3 8 6 0.03 0.04 4"), std::invalid_argument);
}

TEST(TargetTest, DetectsGeneratedArucoWithPairedCorrespondences) {
  const PatternConfig config = parsePattern("1 4 7 0.04");
  cv::Mat marker;
  cv::aruco::drawMarker(
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), 7, 320,
      marker, 1);
  cv::Mat image(500, 500, CV_8UC1, cv::Scalar(255));
  marker.copyTo(image(cv::Rect(90, 90, marker.cols, marker.rows)));

  const auto observation = TargetDetector(config).detect(image);

  ASSERT_TRUE(observation.has_value());
  EXPECT_EQ(observation->imagePoints.size(), 4U);
  EXPECT_EQ(observation->imagePoints.size(), observation->objectPoints.size());
}

TEST(TargetTest, CanonicalArucoMarkerMatchesDictionaryAndCoordinates) {
  for (const int id : {49, 50, 99, 100, 249, 250, 999}) {
    const PatternConfig config =
        parsePattern("1 4 " + std::to_string(id) + " 0.04");
    const cv::Mat modules = canonicalArucoMarker(config);
    ASSERT_EQ(modules.rows, 6);
    ASSERT_EQ(modules.cols, 6);
    ASSERT_EQ(modules.type(), CV_8UC1);
    EXPECT_EQ(arucoDictionaryCapacity(config), id < 50    ? 50
                                               : id < 100 ? 100
                                               : id < 250 ? 250
                                                          : 1000);
    for (int index = 0; index < modules.rows; ++index) {
      EXPECT_EQ(modules.at<std::uint8_t>(0, index), 0);
      EXPECT_EQ(modules.at<std::uint8_t>(modules.rows - 1, index), 0);
      EXPECT_EQ(modules.at<std::uint8_t>(index, 0), 0);
      EXPECT_EQ(modules.at<std::uint8_t>(index, modules.cols - 1), 0);
    }

    cv::Mat enlarged;
    cv::resize(modules, enlarged, cv::Size(300, 300), 0.0, 0.0,
               cv::INTER_NEAREST);
    cv::Mat image(420, 420, CV_8UC1, cv::Scalar(255));
    enlarged.copyTo(image(cv::Rect(60, 60, 300, 300)));
    const auto observation = TargetDetector(config).detect(image);
    ASSERT_TRUE(observation);
    ASSERT_EQ(observation->objectPoints.size(), 4U);
    const float half = static_cast<float>(config.markerSizeMeters / 2.0);
    EXPECT_EQ(observation->objectPoints[0], cv::Point3f(-half, half, 0.0F));
    EXPECT_EQ(observation->objectPoints[1], cv::Point3f(half, half, 0.0F));
    EXPECT_EQ(observation->objectPoints[2], cv::Point3f(half, -half, 0.0F));
    EXPECT_EQ(observation->objectPoints[3], cv::Point3f(-half, -half, 0.0F));
  }

  EXPECT_THROW(canonicalArucoMarker(parsePattern("2 6 3 0.028")),
               std::invalid_argument);
}

TEST(TargetTest, DetectsGeneratedChessboard) {
  constexpr int rows = 6;
  constexpr int columns = 9;
  constexpr int square = 48;
  cv::Mat image((rows + 1) * square, (columns + 1) * square, CV_8UC1,
                cv::Scalar(255));
  for (int row = 0; row <= rows; ++row) {
    for (int column = 0; column <= columns; ++column) {
      if ((row + column) % 2 == 0) {
        cv::rectangle(image,
                      cv::Rect(column * square, row * square, square, square),
                      cv::Scalar(0), cv::FILLED);
      }
    }
  }

  const auto observation =
      TargetDetector(parsePattern("2 6 9 0.025")).detect(image);

  ASSERT_TRUE(observation.has_value());
  EXPECT_EQ(observation->imagePoints.size(), 54U);
  EXPECT_EQ(observation->imagePoints.size(), observation->objectPoints.size());
}

TEST(TargetTest, DetectsGeneratedCharucoWithMatchedIds) {
  const PatternConfig config = parsePattern("3 8 6 0.04 0.03 4");
  const auto dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  const auto board = cv::aruco::CharucoBoard::create(
      config.columns, config.rows, static_cast<float>(config.squareSizeMeters),
      static_cast<float>(config.markerSizeMeters), dictionary);
  cv::Mat image;
  board->draw(cv::Size(700, 900), image, 30, 1);

  const auto observation = TargetDetector(config).detect(image);

  ASSERT_TRUE(observation.has_value());
  EXPECT_GE(observation->imagePoints.size(), 4U);
  EXPECT_EQ(observation->imagePoints.size(), observation->objectPoints.size());
}

TEST(TargetTest, ArucoPoseRequiresKnownIntrinsics) {
  const PatternConfig config = parsePattern("1 4 7 0.04");
  TargetObservation observation;
  observation.objectPoints = {{-0.02F, 0.02F, 0.0F},
                              {0.02F, 0.02F, 0.0F},
                              {0.02F, -0.02F, 0.0F},
                              {-0.02F, -0.02F, 0.0F}};
  observation.imagePoints.resize(4);

  EXPECT_THROW(
      TargetPoseEstimator::estimate(config, {observation}, cv::Size(640, 480)),
      std::invalid_argument);
}

TEST(TargetTest, EstimatesKnownIntrinsicsPoseAndFiltersBadView) {
  const PatternConfig config = parsePattern("2 2 2 0.04");
  TargetObservation good;
  good.objectPoints = {{0.0F, 0.0F, 0.0F},
                       {0.04F, 0.0F, 0.0F},
                       {0.0F, 0.04F, 0.0F},
                       {0.04F, 0.04F, 0.0F}};
  const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 620.0, 0.0, 320.0,
                                0.0, 615.0, 240.0, 0.0, 0.0, 1.0);
  const cv::Mat rotationVector = (cv::Mat_<double>(3, 1) << 0.25, -0.15, 0.08);
  const cv::Mat translationVector =
      (cv::Mat_<double>(3, 1) << -0.02, 0.01, 0.55);
  cv::projectPoints(good.objectPoints, rotationVector, translationVector,
                    cameraMatrix, cv::Mat(), good.imagePoints);
  TargetObservation bad = good;
  bad.imagePoints[0] += cv::Point2f(30.0F, -25.0F);
  TargetPoseOptions options;
  options.maximumReprojectionErrorPixels = 0.5;

  const TargetEstimationResult result =
      TargetPoseEstimator::estimate(config, {good, bad}, cv::Size(640, 480),
                                    cameraMatrix, cv::Mat(), options);

  ASSERT_EQ(result.poses.size(), 2U);
  EXPECT_TRUE(result.poses[0].has_value());
  EXPECT_FALSE(result.poses[1].has_value());
  EXPECT_EQ(result.rejectedViewIndices, std::vector<std::size_t>{1});

  cv::Mat invalidCameraMatrix = cameraMatrix.clone();
  invalidCameraMatrix.at<double>(0, 0) = 0.0;
  EXPECT_THROW(TargetPoseEstimator::estimate(config, {good}, cv::Size(640, 480),
                                             invalidCameraMatrix, cv::Mat()),
               std::invalid_argument);

  const cv::Mat invalidDistortion = (cv::Mat_<double>(1, 3) << 0.0, 0.0, 0.0);
  EXPECT_THROW(TargetPoseEstimator::estimate(config, {good}, cv::Size(640, 480),
                                             cameraMatrix, invalidDistortion),
               std::invalid_argument);
}

TEST(TargetTest, IntrinsicEstimationRequiresEightViewsByDefault) {
  const PatternConfig config = parsePattern("2 2 2 0.04");
  TargetObservation observation;
  observation.objectPoints.resize(4);
  observation.imagePoints.resize(4);

  EXPECT_THROW(
      TargetPoseEstimator::estimate(config, {observation}, cv::Size(640, 480)),
      std::invalid_argument);
}

} // namespace
} // namespace hand_eye::core
