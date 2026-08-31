#include "lib/io/calibration_io.hpp"
#include "lib/io/dataset_repository.hpp"

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <unordered_set>

namespace hand_eye::io {
namespace {

namespace fs = std::filesystem;

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    static std::atomic<std::uint64_t> counter{0};
    const auto ticks =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = fs::temp_directory_path() /
            ("hand_eye_io_test_" + std::to_string(ticks) + "_" +
             std::to_string(counter.fetch_add(1)));
    fs::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const fs::path &path() const { return path_; }

private:
  fs::path path_;
};

core::RigidTransform transform(const double x, const double y, const double z,
                               const double roll, const double pitch,
                               const double yaw) {
  const cv::Mat rotationVector = (cv::Mat_<double>(3, 1) << roll, pitch, yaw);
  cv::Mat rotation;
  cv::Rodrigues(rotationVector, rotation);
  const cv::Mat translation = (cv::Mat_<double>(3, 1) << x, y, z);
  return {rotation, translation};
}

cv::Mat testImage(const std::uint8_t value = 127) {
  return cv::Mat(32, 48, CV_8UC3, cv::Scalar(value, value / 2, 255 - value))
      .clone();
}

void expectTransformNear(const core::RigidTransform &actual,
                         const core::RigidTransform &expected,
                         const double tolerance = 1e-9) {
  EXPECT_LE(cv::norm(actual.translation - expected.translation), tolerance);
  EXPECT_LE(cv::norm(actual.rotation - expected.rotation), tolerance);
}

TEST(DatasetRepositoryTest, PreparesEmptyDatasetAndResumesExplicitly) {
  TemporaryDirectory temporary;
  const fs::path dataset = temporary.path() / "dataset";
  DatasetRepository repository(dataset);

  const DatasetPreparation created =
      repository.prepare(false, false, PoseFormat::QuaternionXyzw);
  EXPECT_TRUE(created.created);
  EXPECT_EQ(created.nextSampleIndex, 0U);
  EXPECT_TRUE(fs::is_directory(dataset / kImagesDirectory));
  EXPECT_TRUE(fs::is_regular_file(dataset / kPosesFilename));
  EXPECT_TRUE(fs::is_regular_file(dataset / kSamplesFilename));

  std::string error;
  EXPECT_TRUE(repository.validate(PoseFormat::QuaternionXyzw, &error, false));
  EXPECT_TRUE(error.empty());
  EXPECT_FALSE(repository.validate(PoseFormat::QuaternionXyzw, &error));
  EXPECT_EQ(error, "dataset contains no samples");
  EXPECT_EQ(repository.nextSampleIndex(PoseFormat::QuaternionXyzw), 0U);

  const DatasetPreparation resumed =
      repository.prepare(true, false, PoseFormat::QuaternionXyzw);
  EXPECT_FALSE(resumed.created);
  EXPECT_EQ(resumed.nextSampleIndex, 0U);
  EXPECT_THROW(repository.prepare(false, false, PoseFormat::QuaternionXyzw),
               std::runtime_error);
  EXPECT_THROW(repository.prepare(true, true, PoseFormat::QuaternionXyzw),
               std::invalid_argument);
}

TEST(DatasetRepositoryTest, AppendsInspectsFiltersAndRemovesSamples) {
  TemporaryDirectory temporary;
  DatasetRepository repository(temporary.path() / "dataset");
  repository.prepare(false, false, PoseFormat::QuaternionXyzw);

  const core::RigidTransform first = transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  const core::RigidTransform second =
      transform(0.10, -0.20, 0.30, 0.0, 0.0, std::numbers::pi / 2.0);
  EXPECT_EQ(repository.appendSample(
                testImage(80), encodePose(first, PoseFormat::QuaternionXyzw),
                SampleAudit{1'000'000'000, 995'000'000},
                PoseFormat::QuaternionXyzw),
            0U);
  EXPECT_EQ(repository.appendSample(
                testImage(160), encodePose(second, PoseFormat::QuaternionXyzw),
                SampleAudit{2'000'000'000, 2'012'000'000},
                PoseFormat::QuaternionXyzw),
            1U);

  const DatasetInspection inspection =
      repository.inspect(PoseFormat::QuaternionXyzw);
  ASSERT_TRUE(inspection.valid) << inspection.errorMessage();
  EXPECT_EQ(inspection.nextSampleIndex, 2U);
  EXPECT_EQ(inspection.statistics.imageCount, 2U);
  EXPECT_EQ(inspection.statistics.poseRowCount, 2U);
  EXPECT_EQ(inspection.statistics.validPoseRowCount, 2U);
  EXPECT_NEAR(inspection.statistics.translationRangeM,
              std::sqrt(0.10 * 0.10 + 0.20 * 0.20 + 0.30 * 0.30), 1e-9);
  EXPECT_GT(inspection.statistics.rotationSpreadDeg, 80.0);

  const auto rows = repository.acceptedPoseRows(PoseFormat::QuaternionXyzw);
  ASSERT_EQ(rows.size(), 2U);
  expectTransformNear(rows[0].transform, first);
  expectTransformNear(rows[1].transform, second);

  const auto filtered = repository.loadRobotTransforms(
      PoseFormat::QuaternionXyzw, std::unordered_set<std::size_t>{0});
  ASSERT_EQ(filtered.size(), 1U);
  expectTransformNear(filtered.front(), second);

  repository.removeLastSample(PoseFormat::QuaternionXyzw);
  const DatasetInspection afterRemoval =
      repository.inspect(PoseFormat::QuaternionXyzw);
  ASSERT_TRUE(afterRemoval.valid) << afterRemoval.errorMessage();
  EXPECT_EQ(afterRemoval.nextSampleIndex, 1U);
  EXPECT_FALSE(fs::exists(repository.path() / kImagesDirectory / "1.png"));
  ASSERT_EQ(repository.poseRows(PoseFormat::QuaternionXyzw).size(), 1U);
  expectTransformNear(
      repository.poseRows(PoseFormat::QuaternionXyzw).front().transform, first);
}

TEST(DatasetRepositoryTest, RoundTripsEverySupportedPoseFormat) {
  TemporaryDirectory temporary;
  const core::RigidTransform expected =
      transform(0.14, -0.08, 0.32, 0.2, -0.3, 0.4);
  for (std::int32_t value = 1; value <= 6; ++value) {
    const PoseFormat format = parsePoseFormat(value);
    DatasetRepository repository(temporary.path() /
                                 ("format_" + std::to_string(value)));
    repository.prepare(false, false, format);
    repository.appendSample(testImage(), encodePose(expected, format), {},
                            format);
    const auto loaded = repository.loadRobotTransforms(format);
    ASSERT_EQ(loaded.size(), 1U) << "pose format " << value;
    expectTransformNear(loaded.front(), expected, 1e-8);
  }
  EXPECT_THROW(parsePoseFormat(0), std::invalid_argument);
  EXPECT_THROW(parsePoseFormat(7), std::invalid_argument);
}

TEST(DatasetRepositoryTest, DetectsNonContiguousImagesAndInvalidPoseRows) {
  TemporaryDirectory temporary;
  DatasetRepository repository(temporary.path() / "dataset");
  repository.prepare(false, false, PoseFormat::QuaternionXyzw);
  const auto pose =
      encodePose(core::RigidTransform::identity(), PoseFormat::QuaternionXyzw);
  repository.appendSample(testImage(10), pose, {}, PoseFormat::QuaternionXyzw);
  repository.appendSample(testImage(20), pose, {}, PoseFormat::QuaternionXyzw);
  repository.appendSample(testImage(30), pose, {}, PoseFormat::QuaternionXyzw);

  fs::remove(repository.path() / kImagesDirectory / "1.png");
  DatasetInspection inspection = repository.inspect(PoseFormat::QuaternionXyzw);
  EXPECT_FALSE(inspection.valid);
  EXPECT_NE(inspection.errorMessage().find("not contiguous"),
            std::string::npos);

  repository.prepare(false, true, PoseFormat::QuaternionXyzw);
  repository.appendSample(testImage(10), pose, {}, PoseFormat::QuaternionXyzw);
  fs::copy_file(repository.path() / kImagesDirectory / "0.png",
                repository.path() / kImagesDirectory / "00.png");
  inspection = repository.inspect(PoseFormat::QuaternionXyzw);
  EXPECT_FALSE(inspection.valid);
  EXPECT_NE(inspection.errorMessage().find("image name is not canonical"),
            std::string::npos);

  repository.prepare(false, true, PoseFormat::QuaternionXyzw);
  ASSERT_TRUE(cv::imwrite(
      (repository.path() / kImagesDirectory / "0.png").string(), testImage()));
  {
    std::ofstream poses(repository.path() / kPosesFilename, std::ios::trunc);
    poses << "this\tis\tnot\ta\tpose\n";
  }
  inspection = repository.inspect(PoseFormat::QuaternionXyzw);
  EXPECT_FALSE(inspection.valid);
  EXPECT_NE(inspection.errorMessage().find("invalid pose row"),
            std::string::npos);
}

TEST(DatasetRepositoryTest, DetectsUndecodablePngImages) {
  TemporaryDirectory temporary;
  DatasetRepository repository(temporary.path() / "dataset");
  repository.prepare(false, false, PoseFormat::QuaternionXyzw);
  repository.appendSample(
      testImage(),
      encodePose(core::RigidTransform::identity(), PoseFormat::QuaternionXyzw),
      {}, PoseFormat::QuaternionXyzw);

  {
    std::ofstream corrupt(repository.path() / kImagesDirectory / "0.png",
                          std::ios::binary | std::ios::trunc);
    corrupt << "not a PNG";
  }

  const DatasetInspection inspection =
      repository.inspect(PoseFormat::QuaternionXyzw);
  EXPECT_FALSE(inspection.valid);
  EXPECT_NE(inspection.errorMessage().find("cannot decode image: 0.png"),
            std::string::npos);
}

TEST(DatasetRepositoryTest, RejectsDangerousDatasetPaths) {
  TemporaryDirectory temporary;

  EXPECT_THROW(DatasetRepository("."), std::invalid_argument);
  EXPECT_THROW(DatasetRepository(fs::current_path().parent_path()),
               std::invalid_argument);
  EXPECT_THROW(DatasetRepository(fs::temp_directory_path()),
               std::invalid_argument);
  EXPECT_NO_THROW(DatasetRepository(temporary.path() / "dataset"));
}

TEST(DatasetRepositoryTest, FailedAppendLeavesPreviousDatasetUntouched) {
  TemporaryDirectory temporary;
  DatasetRepository repository(temporary.path() / "dataset");
  repository.prepare(false, false, PoseFormat::QuaternionXyzw);
  const auto pose =
      encodePose(core::RigidTransform::identity(), PoseFormat::QuaternionXyzw);
  repository.appendSample(testImage(), pose, {}, PoseFormat::QuaternionXyzw);

  const std::string posesBefore = [&] {
    std::ifstream input(repository.path() / kPosesFilename);
    return std::string(std::istreambuf_iterator<char>(input), {});
  }();
  const cv::Mat unsupportedChannels(8, 8, CV_8UC2, cv::Scalar(1, 2));
  EXPECT_THROW(repository.appendSample(unsupportedChannels, pose, {},
                                       PoseFormat::QuaternionXyzw),
               cv::Exception);

  std::ifstream input(repository.path() / kPosesFilename);
  const std::string posesAfter(std::istreambuf_iterator<char>(input), {});
  EXPECT_EQ(posesAfter, posesBefore);
  EXPECT_EQ(repository.nextSampleIndex(PoseFormat::QuaternionXyzw), 1U);
  EXPECT_FALSE(fs::exists(repository.path() / kImagesDirectory / "1.png"));
}

TEST(DatasetRepositoryTest, ResumesLegacyDatasetWithoutAuditFile) {
  TemporaryDirectory temporary;
  DatasetRepository repository(temporary.path() / "dataset");
  repository.prepare(false, false, PoseFormat::QuaternionXyzw);
  const auto pose =
      encodePose(core::RigidTransform::identity(), PoseFormat::QuaternionXyzw);
  repository.appendSample(testImage(), pose, {}, PoseFormat::QuaternionXyzw);
  {
    std::ifstream posesInput(repository.path() / kPosesFilename);
    std::string poseText(std::istreambuf_iterator<char>(posesInput), {});
    while (!poseText.empty() && poseText.back() == '\n') {
      poseText.pop_back();
    }
    std::ofstream posesOutput(repository.path() / kPosesFilename,
                              std::ios::trunc);
    posesOutput << poseText << "\t42\t123.5\n";
  }
  fs::remove(repository.path() / kSamplesFilename);

  const DatasetPreparation resumed =
      repository.prepare(true, false, PoseFormat::QuaternionXyzw);
  EXPECT_EQ(resumed.nextSampleIndex, 1U);
  const auto legacyRows = repository.poseRows(PoseFormat::QuaternionXyzw);
  ASSERT_EQ(legacyRows.size(), 1U);
  ASSERT_EQ(legacyRows.front().values.size(), 9U);
  EXPECT_DOUBLE_EQ(legacyRows.front().values[7], 42.0);
  EXPECT_DOUBLE_EQ(legacyRows.front().values[8], 123.5);
  repository.appendSample(testImage(200), pose,
                          SampleAudit{20'000'000, 19'000'000},
                          PoseFormat::QuaternionXyzw);
  ASSERT_TRUE(repository.inspect(PoseFormat::QuaternionXyzw).valid);

  std::ifstream audit(repository.path() / kSamplesFilename);
  std::string line;
  std::size_t lines = 0;
  while (std::getline(audit, line)) {
    if (!line.empty()) {
      ++lines;
    }
  }
  EXPECT_EQ(lines, 3U); // header, legacy placeholder, new audit row
}

TEST(CalibrationIOTest, RoundTripsCalibrationAndReadsIntrinsics) {
  TemporaryDirectory temporary;
  const fs::path output = temporary.path() / "nested" / "calibration.yaml";

  CalibrationDocument expected;
  expected.transform = transform(0.1, -0.2, 0.3, 0.1, 0.2, -0.1);
  expected.cameraMatrix = (cv::Mat_<double>(3, 3) << 520.0, 0.0, 320.0, 0.0,
                           515.0, 240.0, 0.0, 0.0, 1.0);
  expected.distortionCoefficients =
      (cv::Mat_<double>(1, 5) << 0.1, -0.05, 0.0, 0.0, 0.01);
  expected.method = "park";
  expected.calibrationType = "eye_in_hand";
  expected.parentFrame = "tool0";
  expected.childFrame = "camera";
  expected.translationRmsM = 0.0012;
  expected.rotationRmsDeg = 0.34;
  expected.rejectedSampleIndices = {2, 7, 11};

  CalibrationIO::write(output, expected);
  const CalibrationDocument actual = CalibrationIO::read(output);
  expectTransformNear(actual.transform, expected.transform);
  EXPECT_LE(cv::norm(actual.cameraMatrix - expected.cameraMatrix), 1e-12);
  EXPECT_LE(
      cv::norm(actual.distortionCoefficients - expected.distortionCoefficients),
      1e-12);
  EXPECT_EQ(actual.method, expected.method);
  EXPECT_EQ(actual.calibrationType, expected.calibrationType);
  EXPECT_EQ(actual.parentFrame, expected.parentFrame);
  EXPECT_EQ(actual.childFrame, expected.childFrame);
  EXPECT_DOUBLE_EQ(actual.translationRmsM, expected.translationRmsM);
  EXPECT_DOUBLE_EQ(actual.rotationRmsDeg, expected.rotationRmsDeg);
  EXPECT_EQ(actual.rejectedSampleIndices, expected.rejectedSampleIndices);

  const CameraIntrinsics intrinsics = CalibrationIO::readIntrinsics(output);
  EXPECT_LE(cv::norm(intrinsics.cameraMatrix - expected.cameraMatrix), 1e-12);
  EXPECT_LE(cv::norm(intrinsics.distortionCoefficients -
                     expected.distortionCoefficients),
            1e-12);
}

TEST(CalibrationIOTest, ReadsRosParameterIntrinsicsYaml) {
  TemporaryDirectory temporary;
  const fs::path input = temporary.path() / "realsense_intrinsics.yaml";
  {
    std::ofstream stream(input);
    stream << "/**:\n"
              "  ros__parameters:\n"
              "    width: 960\n"
              "    height: 540\n"
              "    K: [700.0876172, 0.0, 479.392501, 0.0, 698.7540746, "
              "269.4859403, 0.0, 0.0, 1.0]\n"
              "    D: [0.1975899012, -0.5307554581, -0.00695480994, "
              "0.004085926974, 0.3689778213]\n";
  }

  const CameraIntrinsics intrinsics = CalibrationIO::readIntrinsics(input);
  ASSERT_EQ(intrinsics.cameraMatrix.rows, 3);
  ASSERT_EQ(intrinsics.cameraMatrix.cols, 3);
  ASSERT_EQ(intrinsics.distortionCoefficients.rows, 1);
  ASSERT_EQ(intrinsics.distortionCoefficients.cols, 5);
  EXPECT_EQ(intrinsics.width, 960);
  EXPECT_EQ(intrinsics.height, 540);
  EXPECT_DOUBLE_EQ(intrinsics.cameraMatrix.at<double>(0, 0), 700.0876172);
  EXPECT_DOUBLE_EQ(intrinsics.cameraMatrix.at<double>(0, 2), 479.392501);
  EXPECT_DOUBLE_EQ(intrinsics.cameraMatrix.at<double>(1, 1), 698.7540746);
  EXPECT_DOUBLE_EQ(intrinsics.distortionCoefficients.at<double>(0, 4),
                   0.3689778213);
}

} // namespace
} // namespace hand_eye::io
