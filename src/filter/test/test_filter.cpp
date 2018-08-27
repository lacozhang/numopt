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
