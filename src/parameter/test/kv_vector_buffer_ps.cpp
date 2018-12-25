/*
 * =====================================================================================
 *
 *       Filename:  kv_vector_buffer_ps.cpp
 *
 *    Description:  kv vector with buffer enabled
 *
 *        Version:  1.0
 *        Created:  12/24/2018 21:34:28
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
const int kChannelId = 4;
DArray<KeyType> kAllKeys = DArray<KeyType>{0, 1, 3, 4, 5, 8};
DArray<KeyType> kCh0Keys = DArray<KeyType>{0, 3, 5, 8};
DArray<KeyType> kCh1Keys = DArray<KeyType>{1, 3, 4, 5};
DArray<ValType> kCh0Params = DArray<ValType>{1, 2, 1, 2, 1, 2, 1, 2};
DArray<ValType> kCh1Params = DArray<ValType>{3, 4, 3, 4, 3, 4, 3, 4};
} // namespace

class Server : public App {
public:
  Server() : param_(true, 2) {
    param_[kChannelId].key_ = kAllKeys;
    LOG(INFO) << MyNodeID() << " : server parameter table with id "
              << param_.id();
  }

  virtual void run() override {
    sys_.manager().waitWorkersReady();
    LOG(INFO) << MyNode().DebugString();
    int ts = 0;
    param_.waitReceivedRequest(ts, kWorkerGroup);
    auto recv = param_.buffer(ts);
    if (!recv.values_.empty()) {
      LOG(INFO) << "server channel: " << recv.channel_;
      LOG(INFO) << "server range  : " << recv.idxRange_.begin() << " / "
                << recv.idxRange_.end();
      LOG(INFO) << "server values : " << recv.values_.size();
      param_[kChannelId].val_ = recv.values_[0];
      LOG(INFO) << "more assignment";
      LOG(INFO) << "lvalue " << param_[kChannelId].val_;
      LOG(INFO) << "rvalue " << recv.values_[1];
      param_[kChannelId].val_.vec() += recv.values_[1].vec();
    }
    param_.finishReceivedRequest(ts + 1, kWorkerGroup);
  }

private:
  KVVector<KeyType, ValType> param_;
};

class Worker : public App {
public:
  Worker() : param_(false, 2) {
    LOG(INFO) << MyNodeID() << ": worker parameter table with id "
              << param_.id();
    param_[kChannelId].key_.copyFrom(kAllKeys);
  }

  virtual void run() override {
    LOG(INFO) << MyNodeID() << ": worker node " << MyRank() << std::endl;

    DArray<KeyType> key;
    LOG(INFO) << MyNode().DebugString();

    if (MyRank() == 0) {
      key = kCh0Keys;
    } else {
      key = kCh1Keys;
    }

    int ts = param_.push(Parameter::request(kChannelId), key,
                         {kCh0Params, kCh1Params});
    LOG(INFO) << "send push request ts " << ts;

    // a request depend on virtual request {ts+1}, which will be marked as
    // finished by server.
    param_.wait(
        param_.pull(Parameter::request(kChannelId, ts + 2, {ts + 1}), key));

    LOG(INFO) << MyNodeID() << ": pulled value from channel " << kChannelId
              << " " << param_[kChannelId].val_;
  }

private:
  KVVector<KeyType, ValType> param_;
};

App *App::Create(const std::string &conf) {
  if (IsWorker()) {
    return new Worker();
  }
  if (IsServer()) {
    return new Server();
  }

  return new App();
}
} // namespace mltools

int main(int argc, char *argv[]) { return mltools::RunSystem(argc, argv); }
