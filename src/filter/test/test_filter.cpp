/*
 * =====================================================================================
 *
 *       Filename:  test_filter.cpp
 *
 *    Description:  unit test files for filter
 *
 *        Version:  1.0
 *        Created:  07/19/2018 08:56:16
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "filter/compressing.h"
#include "filter/fixing_float.h"
#include "util/dynamic_array_impl.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mltools;

DArray<int32> testKeys;
DArray<float> testValues;

void createMessage(Message *msg, bool hasKey) {
  if (hasKey) {
    msg->key_ = testKeys;
  }
  msg->value_.resize(1);
  msg->value_[0] = testValues;
  msg->add_filter(FilterConfig::COMPRESSING);
}

TEST(Filter, compression) {
  testKeys.resize(10000, 0);
  testValues.resize(10000, 0);
  for (int i = 0; i < 10000; ++i) {
    testKeys[i] = i + 10;
    testValues[i] = i - 5.0;
  }
  Message msg;
  createMessage(&msg, true);
  LOG(INFO) << "Message before encode" << std::endl << msg.DebugString();
  CompressingFilter filter;
  filter.encode(&msg);
  LOG(INFO) << "Encode finished";
  LOG(INFO) << "Message after decode" << std::endl << msg.DebugString();
  filter.decode(&msg);
  DArray<int32> decodedKeys;
  decodedKeys = msg.key_;
  DArray<float> decodedValues;
  decodedValues = msg.value_[0];
  for (int i = 0; i < 10000; ++i) {
    CHECK_EQ(testKeys[i], decodedKeys[i]);
    CHECK_DOUBLE_EQ(testValues[i], decodedValues[i]);
  }
}

TEST(Filter, FixingFloat) {
  std::shared_ptr<Message> msg(new Message());
  auto filterConf = msg->add_filter(FilterConfig::FIXING_FLOAT);
  auto conf = filterConf->add_fixed_point();
  conf->set_min_value(-100);
  conf->set_max_value(100);
  filterConf->set_num_bytes(3);
  double quantizationErrorBound = 1.0 / (1 << (filterConf->num_bytes() * 8));

  DArray<float> ax = {100.0, 0.1, -0.3, -100.0};
  DArray<float> bx = {100.0, .1, -0.2, -100.0};
  LOG(INFO) << "Array before decode " << dbgstr<float>(ax.data(), ax.size());
  LOG(INFO) << "Array before decode " << dbgstr<float>(bx.data(), bx.size());

  msg->add_value(ax);
  msg->add_value(bx);

  FixingFloatFilter filter;
  filter.encode(msg.get());
  filter.decode(msg.get());

  LOG(INFO) << "Decoding result" << DArray<float>(msg->value_[0]);
  LOG(INFO) << "Decoding result" << DArray<float>(msg->value_[0]);

  DArray<float> decoded_ax(msg->value_[0]);
  DArray<float> decoded_bx(msg->value_[1]);

  ASSERT_EQ(decoded_ax.size(), ax.size());
  ASSERT_EQ(decoded_bx.size(), bx.size());
  for (size_t i = 0; i < ax.size(); ++i) {
    ASSERT_LE(std::fabs(decoded_ax[i] - ax[i]), 1e-4);
  }

  for (size_t i = 0; i < bx.size(); ++i) {
    ASSERT_LE(std::fabs(decoded_bx[i] - bx[i]), 1e-4);
  }
}
