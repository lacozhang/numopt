/*
 * =====================================================================================
 *
 *       Filename:  heartbeat_info.cpp
 *
 *    Description:  main implementation
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:27:01
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "system/heartbeat_info.h"
#include <chrono>

namespace mltools {
HeartbeatInfo::HeartbeatInfo() : timers_(static_cast<int>(TimerType::NUM)) {
  inBytes_ = outBytes_ = 0;
}

HeartbeatInfo::~HeartbeatInfo() {}

void HeartbeatInfo::startTimer(const HeartbeatInfo::TimerType type) {
  Lock l(mu_);
  timers_.at(static_cast<int>(type)).start();
}

void HeartbeatInfo::stopTimer(const HeartbeatInfo::TimerType type) {
  Lock l(mu_);
  timers_.at(static_cast<int>(type)).stop();
}

HeartbeatInfo::Snapshot HeartbeatInfo::dump() {
  HeartbeatInfo::Snapshot ret;

  // process cpu
  std::ifstream my_cpu_stat("/proc/self/stat", std::ifstream::in);
  CHECK(my_cpu_stat) << "open /proc/self/stat failed [" << strerror(errno)
                     << "]";
  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  my_cpu_stat >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
      stime >> cutime >> cstime >> priority >> nice;
  my_cpu_stat.close();
  ret.processUser_ = std::stoull(utime);
  ret.processSys_ = std::stoull(stime);

  // host cpu
  std::ifstream host_cpu_stat("/proc/stat", std::ifstream::in);
  CHECK(host_cpu_stat) << "open /proc/stat failed [" << strerror(errno) << "]";
  string label, host_user, host_nice, host_sys, host_idle, host_iowait;
  host_cpu_stat >> label >> host_user >> host_nice >> host_sys >> host_idle >>
      host_iowait;
  host_cpu_stat.close();
  ret.hostUser_ = std::stoull(host_user);
  ret.hostSys_ = std::stoull(host_sys);
  ret.hostCpu_ = std::stoull(host_user) + std::stoull(host_nice) +
                 std::stoull(host_sys) + std::stoull(host_idle) +
                 std::stoull(host_iowait);

  // host network bandwidth usage
  if (!interface_.empty()) {
    std::ifstream host_net_dev_stat("/proc/net/dev", std::ifstream::in);
    CHECK(host_net_dev_stat)
        << "open /proc/net/dev failed [" << strerror(errno) << "]";

    // find interface
    string line;
    bool interface_found = false;
    while (std::getline(host_net_dev_stat, line)) {
      if (std::string::npos != line.find(interface_)) {
        interface_found = true;
        break;
      }
    }
    CHECK(interface_found) << "I cannot find interface[" << interface_
                           << "] in /proc/net/dev";

    // read counters
    string face, r_bytes, r_packets, r_errs, r_drop, r_fifo, r_frame;
    string r_compressed, r_multicast, t_bytes, t_packets;
    std::stringstream ss(line);
    ss >> face >> r_bytes >> r_packets >> r_errs >> r_drop >> r_fifo >>
        r_frame >> r_compressed >> r_multicast >> t_bytes >> t_packets;
    host_net_dev_stat.close();

    ret.hostInBytes_ = std::stoull(r_bytes);
    ret.hostOutBytes_ = std::stoull(t_bytes);
  }

  return ret;
}

HeartbeatReport HeartbeatInfo::get() {
  Lock l(mu_);
  HeartbeatReport report;
  HeartbeatInfo::Snapshot snapshot_now = dump();

  // interval between invocations of get()
  double total_milli = totalTimer_.stop();
  if (total_milli < 1.0) {
    total_milli = 1.0;
  }

  report.set_hostname(hostname_);

  report.set_seconds_since_epoch(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  report.set_total_time_milli(total_milli);
  report.set_busy_time_milli(
      timers_.at(static_cast<size_t>(HeartbeatInfo::TimerType::BUSY)).get());

  report.set_net_in_mb(inBytes_ / 1024 / 1024);
  inBytes_ = 0;
  report.set_net_out_mb(outBytes_ / 1024 / 1024);
  outBytes_ = 0;

  uint32 process_now = snapshot_now.processUser_ + snapshot_now.processSys_;
  uint32 process_last = last_.processUser_ + last_.processSys_;
  report.set_process_cpu_usage(cpuCoreNumbers_ * 100 *
                               static_cast<float>(process_now - process_last) /
                               (snapshot_now.hostCpu_ - last_.hostCpu_));

  uint32 host_now = snapshot_now.hostUser_ + snapshot_now.hostSys_;
  uint32 host_last = last_.hostUser_ + last_.hostSys_;
  report.set_host_cpu_usage(cpuCoreNumbers_ * 100 *
                            static_cast<float>(host_now - host_last) /
                            (snapshot_now.hostCpu_ - last_.hostCpu_));

  report.set_process_rss_mb(ResUsage::myPhyMem());
  report.set_process_virt_mb(ResUsage::myVirMem());
  report.set_host_in_use_gb(ResUsage::hostInUseMem() / 1024);
  report.set_host_in_use_percentage(100 * ResUsage::hostInUseMem() /
                                    ResUsage::hostTotalMem());

  report.set_host_net_in_bw(
      static_cast<uint32>((snapshot_now.hostInBytes_ - last_.hostInBytes_) /
                          (total_milli / 1e3) / 1024 / 1024));
  report.set_host_net_out_bw(
      static_cast<uint32>((snapshot_now.hostOutBytes_ - last_.hostOutBytes_) /
                          (total_milli / 1e3) / 1024 / 1024));

  // reset all timers
  for (auto &timer : timers_) {
    timer.reset();
    timer.start();
  }
  totalTimer_.reset();
  totalTimer_.start();

  last_ = snapshot_now;
  return report;
}
  
  void HeartbeatInfo::init(const string& interface, const string& hostname) {
    interface_ = interface;
    hostname_ = hostname;
    
    // get cpu core number
    char buffer[1024];
    FILE *fp_pipe = popen("grep 'processor' /proc/cpuinfo | wc -l", "r");
    CHECK(nullptr != fp_pipe);
    CHECK(nullptr != fgets(buffer, sizeof(buffer), fp_pipe));
    string core_str(buffer);
    core_str.resize(core_str.size() - 1);
    cpuCoreNumbers_ = std::stoul(core_str);
    pclose(fp_pipe);
    
    // initialize internal status
    get();
  }
} // namespace mltools
