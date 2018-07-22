/*
 * =====================================================================================
 *
 *       Filename:  monitor.h
 *
 *    Description:  A distributed monitor
 *
 *        Version:  1.0
 *        Created:  07/22/2018 14:38:23
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "system/customer.h"
#include "system/sysutil.h"

namespace mltools {

/**
 * @brief The master of slaves, collect information.
 */
template <typename Progress> class MonitorMaster : public Customer {
public:
  MonitorMaster(int id = NextCustomerID()) : Customer(id) {}

  typedef std::function<void(double, std::unordered_map<NodeID, Progress> *)>
      Printer;

  /// @brief set the printer.
  void setPrinter(double timeInterval, Printer printer) {
    timer_.start();
    printer_ = printer;
    interval_ = timeInterval;
  }

  typedef std::function<void(const Progress &, Progress *)> Merger;
  void setMerger(Merger merger) { merger_ = merger; }

  virtual void ProcessRequest(Messager *request) {
    NodeID sender = request->sender_;
    Progress prog;
    CHECK(prog.ParseFromString(request->task_.msg()));
    if (merger_) {
      merger_(prog, &progress_[sender]);
    } else {
      progress_[sender] = prog;
    }

    double time = time_.stop();
    if (time > interval_ && printer_) {
      totalTime_ += time;
      printer_(totalTime_, &progress_);
      timer_.restart();
    } else {
      timer_.start();
    }
  }

private:
  std::unordered_map<NodeID, Progress> progress_;
  double interval_;
  Timer timer_;
  double totalTime_ = 0;
  Merger merger_;
  Printer printer_;
};

template <typename Progress> class MonitorSlaver : public Customer {
public:
  MonitorSlaver(const NodeID &master, int id = NextCustomerID())
      : Customer(id), master_(master) {}
  ~MonitorSlaver() {}

  void report(const Progress &prog) {
    std::string str;
    CHECK(prog.SerializeToString(&str));
    Task report;
    report.set_msg(str);
    submit(report, master_);
  }

protected:
  NodeID master_;
};
} // namespace mltools
