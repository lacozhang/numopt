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

#include "proto/dataconfig.pb.h"
#include "proto/example.pb.h"

#ifndef __INFO_PARSER_H__
#define __INFO_PARSER_H__

namespace mltools {

extern const int kSlotIdMax = 4096;

class InfoParser {
public:
  InfoParser();
  ~InfoParser();
  bool add(Example &ex);
  bool clear();
  const ExampleInfo &info() const;

private:
  size_t num_ex_;
  ExampleInfo info_;
  SlotInfo slotsInfo_[kSlotIdMax];
};
} // namespace mltools

#endif // __INFO_PARSER_H__
