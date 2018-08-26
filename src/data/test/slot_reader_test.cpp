/*
 * =====================================================================================
 *
 *       Filename:  slot_reader_test.cpp
 *
 *    Description:  test SlotReader
 *
 *        Version:  1.0
 *        Created:  07/11/2018 21:46:28
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "data/slot_reader.h"
#include "gtest/gtest.h"
using namespace mltools;
using namespace std;

namespace {
string rcvDataPath = "/Users/edwinzhang/src/parameter_server/example/linear/"
                     "data/rcv1/train/part-00[1-4]";
}

TEST(SlotReader, init) { SlotReader reader; }

TEST(SlotReader, ReadData) {
  DataConfig cfg, cache;
  cfg.set_format(DataConfig::TEXT);
  cfg.set_text(DataConfig::LIBSVM);
  cfg.add_file(rcvDataPath);
  auto res = searchFiles(cfg);
  LOG(INFO) << "read files " << res.DebugString();
  cache.add_file("/tmp/ps_cache");
  ExampleInfo info;
  SlotReader reader(res, cache);
  reader.read(&info);
  LOG(INFO) << "Meta info " << info.DebugString();
}

TEST(SlotReader, ReadDataAgain) {
  DataConfig cfg, cache;
  cfg.set_format(DataConfig::TEXT);
  cfg.set_text(DataConfig::LIBSVM);
  cfg.add_file(rcvDataPath);
  auto res = searchFiles(cfg);
  LOG(INFO) << "read files " << res.DebugString();
  cache.add_file("/tmp/ps_cache");
  ExampleInfo info;
  SlotReader reader(res, cache);
  reader.read(&info);
  LOG(INFO) << "Meta info " << info.DebugString();
}

TEST(SlotReader, ReadSlot) {
  DataConfig cfg, cache;
  cfg.set_format(DataConfig::TEXT);
  cfg.set_text(DataConfig::LIBSVM);
  cfg.add_file(rcvDataPath);
  auto res = searchFiles(cfg);
  cache.add_file("/tmp/ps_cache");
  ExampleInfo info;
  SlotReader reader(res, cache);
  reader.read(&info);
  LOG(INFO) << "Meta info " << info.DebugString();

  for (int i = 1; i < info.slot_size(); ++i) {
    auto slotId = info.slot(i).id();
    auto offset = reader.offset(slotId);
    auto index = reader.index(slotId);
    auto value = reader.value<float>(slotId);
    if (slotId != 0) {
      CHECK_EQ(index.size(), value.size());
    }
    CHECK_EQ(offset.size(), info.num_ex() + 1);
    CHECK_EQ(value.size(), info.slot(slotId).nnz_ele());
  }
}
