#pragma once

#include <charconv>
#include <stdexcept>

namespace parse_helpers {

inline int32_t parseInt(std::string_view value, std::string_view paramName) {
  int32_t parsed = 0;
  const char *begin = value.data();
  const char *end = value.data() + value.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc() || ptr != end) {
    throw std::runtime_error("Error: invalid " + std::string(paramName) + ": " +
                             std::string(value));
  }
  return parsed;
}

inline int32_t parsePositiveInt(std::string_view value,
                                std::string_view paramName) {
  const int32_t parsed = parseInt(value, paramName);
  if (parsed <= 0) {
    throw std::runtime_error("Error: " + std::string(paramName) +
                             " must be positive");
  }
  return parsed;
}

inline double parsePositiveDouble(std::string_view value,
                                  std::string_view paramName) {
  try {
    const std::string text(value);
    size_t parsedChars = 0;
    const double parsed = std::stod(text, &parsedChars);
    if (parsedChars != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    if (parsed <= 0.0) {
      throw std::runtime_error("Error: " + std::string(paramName) +
                               " must be positive");
    }
    return parsed;
  } catch (const std::invalid_argument &) {
    throw std::runtime_error("Error: invalid " + std::string(paramName) + ": " +
                             std::string(value));
  } catch (const std::out_of_range &) {
    throw std::runtime_error("Error: " + std::string(paramName) +
                             " is out of range: " + std::string(value));
  }
}

} // namespace parse_helpers
