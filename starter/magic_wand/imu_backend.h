#ifndef MAGIC_WAND_IMU_BACKEND_H
#define MAGIC_WAND_IMU_BACKEND_H

#if defined(MAGIC_WAND_IMU_BMI270)
#include "src/Arduino_BMI270_BMM150.h"
constexpr const char* kImuProfileName = "nano33ble-sense-rev2-bmi270";
#else
#include <Arduino_LSM9DS1.h>
constexpr const char* kImuProfileName = "nano33ble-sense-original-lsm9ds1";
#endif

// Both Arduino IMU libraries expose board-aligned acceleration in g and
// gyroscope values in degrees per second. Keeping normalization behind this
// function makes the shared coordinate contract explicit and testable.
inline void NormalizeImuAxes(float& x, float& y, float& z) {
  const float normalized_x = x;
  const float normalized_y = y;
  const float normalized_z = z;
  x = normalized_x;
  y = normalized_y;
  z = normalized_z;
}

#endif
