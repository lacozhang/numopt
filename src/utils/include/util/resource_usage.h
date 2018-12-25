/*
 * =====================================================================================
 *
 *       Filename:  resource_usage.h
 *
 *    Description:  utility function to track the resources (time, memory, etc.)
 *
 *        Version:  1.0
 *        Created:  07/18/2018 12:14:34
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include <chrono>
#include <ctime>
#include <ratio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

namespace mltools {
using std::chrono::system_clock;

/// @brief Exact time point now.
static system_clock::time_point tic() { return system_clock::now(); }

/// @brief calculate elapsed time in seconds.
static double toc(system_clock::time_point start) {
  size_t interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        system_clock::now() - start)
                        .count();
  return (interval * 1.0) / 1e3;
}

/// @brief calculate the elapsed time in milliseconds.
static double milliToc(system_clock::time_point start) {
  size_t counts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      system_clock::now() - start)
                      .count();
  return static_cast<double>(counts);
}

/*
static struct timespec hwtic() {
  struct timespec tc;
  clock_gettime(CLOCK_MONOTONIC_RAW, &tc);
  return tc;
}

static double hwtoc(struct timespec start) {
  struct timespec curr;
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr);
  return (double)((curr.tv_sec - start.tv_sec) +
                  (curr.tv_nsec - start.tv_nsec) * 1e-9);
}
*/

class ScopedTimer {
public:
  explicit ScopedTimer(double *aggregateTime) : aggregateTime_(aggregateTime) {
    start_ = tic();
  }

  ~ScopedTimer() { *aggregateTime_ += toc(start_); }

private:
  double *aggregateTime_ = nullptr;
  system_clock::time_point start_;
};

class Timer {
public:
  void start() { tp_ = tic(); }
  void restart() {
    reset();
    start();
  }
  void reset() { time_ = 0; }
  double stop() {
    time_ += toc(tp_);
    return time_;
  }
  double get() { return time_; }
  double getAndRestart() {
    double t = get();
    reset();
    start();
    return t;
  }

private:
  system_clock::time_point tp_ = tic();
  double time_ = 0;
};

class MilliTimer {
public:
  void start() { tp_ = tic(); }
  void restart() {
    reset();
    start();
  }
  void reset() { time_ = 0; }
  double stop() {
    time_ += milliToc(tp_);
    return time_;
  }
  double get() { return time_; }
  double getAndRestart() {
    double t = get();
    reset();
    start();
    return t;
  }

private:
  system_clock::time_point tp_ = tic();
  double time_ = 0;
};

class ResUsage {
public:
  // in Mb
  static double myVirMem() {
    return getLine("/proc/self/status", "VmSize:") / 1e3;
  }
  static double myPhyMem() {
    return getLine("/proc/self/status", "VmRSS:") / 1e3;
  }
  // in Mb
  static double hostInUseMem() {
    return (getLine("/proc/meminfo", "MemTotal:") -
            getLine("/proc/meminfo", "MemFree:") -
            getLine("/proc/meminfo", "Buffers:") -
            getLine("/proc/meminfo", "Cached:")) /
           1024;
  }
  static double hostTotalMem() {
    return getLine("/proc/meminfo", "MemTotal:") / 1024;
  }

private:
  static double getLine(const char *filename, const char *name) {
    FILE *file = fopen(filename, "r");
    char line[128];
    int result = -1;
    while (fgets(line, 128, file) != NULL) {
      if (strncmp(line, name, strlen(name)) == 0) {
        result = parseLine(line);
        break;
      }
    }
    fclose(file);
    return result;
  }

  static int parseLine(char *line) {
    int i = strlen(line);
    while (*line < '0' || *line > '9')
      line++;
    line[i - 3] = '\0';
    i = atoi(line);
    return i;
  }
};
} // namespace mltools
