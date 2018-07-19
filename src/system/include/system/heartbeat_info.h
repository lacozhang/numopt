/*
 * =====================================================================================
 *
 *       Filename:  heartbeat_info.h
 *
 *    Description:  used to monitor the liveness of node
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:26:37
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/heartbeat.pb.h"
#include "util/common.h"
#include "util/resource_usage.h"

namespace mltools {
/// @brief track the liveness of node
class HeartbeatInfo {
public:
  enum class TimerType : unsigned char { BUSY = 0, NUM };

public:
  HeartbeatInfo();
  ~HeartbeatInfo();
  HeartbeatInfo(const HeartbeatInfo &other) = delete;
  HeartbeatInfo &operator=(const HeartbeatInfo &other) = delete;

  HeartbeatReport get();

  /// @brief setup useful information:interface/hostname
  void init(const std::string &interface, const std::string &hostname);
  void startTimer(const HeartbeatInfo::TimerType type);
  void stopTimer(const HeartbeatInfo::TimerType type);

  void incInBytes(const size_t delta) {
    Lock l(mu_);
    inBytes_ += delta;
  }

  void incOutBytes(const size_t delta) {
    Lock l(mu_);
    outBytes_ += delta;
  }

private:
  std::vector<MilliTimer> timers_;
  MilliTimer totalTimer_;

  size_t inBytes_ = 0;
  size_t outBytes_ = 0;

  std::string interface_;
  std::string hostname_;

  /// @brief snapshot of performance counters
  struct Snapshot {
    uint64 processUser_;
    uint64 processSys_;
    uint64 hostUser_;
    uint64 hostSys_;
    uint64 hostCpu_;
    uint64 hostInBytes_;
    uint64 hostOutBytes_;

    Snapshot() {
      processUser_ = processSys_ = 0;
      hostUser_ = hostSys_ = hostCpu_ = 0;
      hostInBytes_ = hostOutBytes_ = 0;
    }

    std::string shortDebugString() {
      std::stringstream ss;
      ss << "{";
      ss << "process_user: " << processUser_ << ", ";
      ss << "process_sys: " << processSys_ << ", ";
      ss << "host_user: " << hostUser_ << ", ";
      ss << "host_sys: " << hostSys_ << ", ";
      ss << "host_cpu: " << hostCpu_ << ", ";
      ss << "host_in_bytes: " << hostInBytes_ << ", ";
      ss << "host_out_bytes: " << hostOutBytes_;
      ss << "}";
      return ss.str();
    }
  };

  HeartbeatInfo::Snapshot last_;
  HeartbeatInfo::Snapshot dump();

  std::mutex mu_;
  size_t cpuCoreNumbers_;
};
} // namespace mltools
