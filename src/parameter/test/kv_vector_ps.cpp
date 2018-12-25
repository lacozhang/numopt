/*
 * =====================================================================================
 *
 *       Filename:  test_parameter.cpp
 *
 *    Description:  general stub for unit test framework
 *
 *        Version:  1.0
 *        Created:  07/22/2018 21:21:00
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "parameter/kv_vector.h"
#include "system/sysutil.h"
#include "util/dynamic_array.h"

namespace mltools {

typedef uint64 KeyType;
typedef double ValType;

namespace {
DArray<KeyType> kCh0Keys = DArray<KeyType>{0, 1, 3, 4, 5};
DArray<KeyType> kCh1Keys = DArray<KeyType>{0, 20, 100, 5000, 5003};
DArray<ValType> kCh0Params = DArray<ValType>{0.2, 2.3, 3.5, 4.2, 5.2939};
DArray<ValType> kCh0Updates = DArray<ValType>{0.5, 2.2, 3.7, 2.2, 100.2};
DArray<ValType> kCh1Params = DArray<ValType>{0.21, 0.3, -8.7, 20.5, 9.9};
DArray<ValType> kCh1Updates = DArray<ValType>{10.21, 102.3, 29.7, 120.5, 19.9};

DArray<ValType> arrayAddition(const DArray<ValType> &v1,
                              const DArray<ValType> &v2) {
  EXPECT_TRUE(v1.size() == v2.size());
  size_t size = v1.size();
  DArray<ValType> ret;
  ret.resize(size);
  ret.setZero();
  for (int i = 0; i < size; ++i) {
    ret[i] = v1[i] + v2[i];
  }
  return ret;
}
} // namespace

class Server : public App {
public:
  Server() : param_() {
    param_[0].key_ = kCh0Keys;
    param_[0].val_ = kCh0Params;

    param_[1].key_ = kCh1Keys;
    param_[1].val_ = kCh1Params;

    LOG(INFO) << MyNodeID() << " : server parameter table with id "
              << param_.id();
  }

private:
  KVVector<KeyType, ValType> param_;
};

class Worker : public App {
public:
  Worker() : param_() {
    LOG(INFO) << MyNodeID() << ": worker parameter table with id "
              << param_.id();
    param_[0].key_.copyFrom(kCh0Keys);
    param_[1].key_.copyFrom(kCh1Keys);
  }

  virtual void run() override {
    LOG(INFO) << MyNodeID() << ": worker node " << MyRank() << std::endl;

    DArray<KeyType> key;
    LOG(INFO) << MyNode().DebugString();

    int ts1 = param_.pull(Parameter::request(0), kCh0Keys);
    int ts2 = param_.pull(Parameter::request(1), kCh1Keys);

    param_.wait(ts1);
    LOG(INFO) << MyNodeID() << ": pulled values from channel 0 "
              << param_[0].val_;
    ASSERT_TRUE(param_[0].val_ == kCh0Params);

    param_.wait(ts2);
    LOG(INFO) << MyNodeID() << ": pulled values from channel 1 "
              << param_[1].val_;
    ASSERT_TRUE(param_[1].val_ == kCh1Params);

    param_[0].val_.clear();
    param_[1].val_.clear();
    DArray<ValType> paramUpdate, param;
    if (MyRank() == 0) {
      param = kCh0Params;
      paramUpdate = kCh0Updates;
      key = kCh0Keys;
    } else {
      param = kCh1Params;
      paramUpdate = kCh1Updates;
      key = kCh1Keys;
    }

    ts1 =
        param_.push(Parameter::request(MyRank(), Message::kInvalidTime, {ts2}),
                    key, {paramUpdate});
    ts2 = param_.pull(
        Parameter::request(MyRank(), Message::kInvalidTime, {ts1}), key);
    param_.wait(ts2);

    auto updatedParams = arrayAddition(param, paramUpdate);
    ASSERT_TRUE(updatedParams == param_[MyRank()].val_)
        << "node rank " << MyNodeID() << ": get vals from server "
        << param_[MyRank()].val_ << "/ calculated values " << updatedParams;
  }

private:
  KVVector<KeyType, ValType> param_;
};

class Scheduler : public App {
  virtual void run() override {
    std::cout << "running from scheduler";
    sys_.manager().waitServersReady();
    sys_.manager().waitWorkersReady();
  }
};

App *App::Create(const std::string &conf) {
  if (IsWorker()) {
    return new Worker();
  }
  if (IsServer()) {
    return new Server();
  }

  if (IsScheduler()) {
    return new Scheduler();
  }

  LOG(FATAL) << "Unknow role type " << MyNode().DebugString();
}
} // namespace mltools

int main(int argc, char *argv[]) { return mltools::RunSystem(argc, argv); }
