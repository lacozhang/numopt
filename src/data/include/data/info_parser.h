/*
 * =====================================================================================
 *
 *       Filename:  info_parser.h
 *
 *    Description:  The main interface for collecting information about training
 * data, key range for each slot, each key group available information in
 * training data
 *
 *        Version:  1.0
 *        Created:  06/10/2018 17:10:18
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "data/common.h"
#include "proto/dataconfig.pb.h"
#include "proto/example.pb.h"
#include "util/integral_types.h"

namespace mltools {

class InfoParser {
public:
  InfoParser();
  ~InfoParser();
  bool add(const Example &ex);
  bool clear();
  const ExampleInfo &info();

private:
  size_t num_ex_;
  ExampleInfo info_;
  SlotInfo slotsInfo_[static_cast<int>(FeatureConstants::kSlotIdMax)];
};
} // namespace mltools
