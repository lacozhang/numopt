/*
 * =====================================================================================
 *
 *       Filename:  kv_map_ps.cpp
 *
 *    Description:  testing key-value map
 *
 *        Version:  1.0
 *        Created:  12/24/2018 21:35:28
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "parameter/kv_map.h"
#include "parameter/kv_vector.h"
#include "system/sysutil.h"

namespace mltools {

typedef uint64 KeyType;
typedef double ValType;

namespace {
const int kChannelId = 4;
DArray<KeyType> kAllKeys = DArray<KeyType>{0, 1, 3, 4, 5, 8};
DArray<KeyType> kCh0Keys = DArray<KeyType>{0, 3, 5, 8};
DArray<KeyType> kCh1Keys = DArray<KeyType>{1, 3, 4, 5};
DArray<ValType> kCh0Params = DArray<ValType>{1.0, 2.0, 3.0, 4.0};
DArray<ValType> kCh1Params = DArray<ValType>{5.0, 6.0, 7.0, 8.0};
} // namespace

struct Entry {
  void get(ValType *data, void *state) { *data = value_; }

  void set(const ValType *data, void *state) { value_ += *data; }

  ValType value_ = 0;
};

class Server : public App {
private:
  KVMap<KeyType, ValType, Entry> param_;
};

class Worker : public App {
public:
  Worker() : param_() {
    LOG(INFO) << MyNodeID() << ": worker parameter table with id "
              << param_.id();
    param_[kChannelId].key_.copyFrom(kAllKeys);
  }

  virtual void run() override {
    LOG(INFO) << MyNodeID() << ": worker node " << MyRank() << std::endl;

    DArray<KeyType> key;
    DArray<ValType> val;
    LOG(INFO) << MyNode().DebugString();

    if (MyRank() == 0) {
      key = kCh0Keys;
      val = kCh0Params;
    } else {
      key = kCh1Keys;
      val = kCh1Params;
    }

    param_.wait(param_.push(Parameter::request(kChannelId), key, {val}));
    param_.wait(param_.pull(Parameter::request(kChannelId), key));
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
