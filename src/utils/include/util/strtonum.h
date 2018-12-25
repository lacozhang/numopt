/*
 * =====================================================================================
 *
 *       Filename:  strtonum.h
 *
 *    Description:  utility function to convert string to different type of
 * values
 *
 *        Version:  1.0
 *        Created:  05/12/2018 19:05:52
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "util/integral_types.h"
#include <stdlib.h>
#include <string>

namespace mltools {
// general expectation is whole string is a number, otherwise return false
// return true if valid

inline bool strtofloat(const char *str, float *val) {
  char *end;
  *val = strtof(str, &end);
  if (*end == '\0')
    return true;
  return false;
}

inline bool strtoi32(const char *str, int32 *val) {
  char *end;
  *val = strtol(str, &end, 10);
  if (*end == '\0')
    return true;
  return false;
}

inline bool strtou64(const char *str, uint64 *val) {
  char *end;
  *val = strtoull(str, &end, 10);
  if (*end == '\0')
    return true;
  return false;
}

inline bool strtofloat(const std::string &str, float *val) {
  return strtofloat(str.c_str(), val);
}

inline bool strtoi32(const std::string &str, int32 *val) {
  return strtoi32(str.c_str(), val);
}

inline bool strtou64(const std::string &str, uint64 *val) {
  return strtou64(str.c_str(), val);
}
} // namespace mltools
