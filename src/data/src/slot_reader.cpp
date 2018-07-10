/*
 * =====================================================================================
 *
 *       Filename:  slot_reader.cpp
 *
 *    Description:  implement coordinate fashion reading
 *
 *        Version:  1.0
 *        Created:  07/08/2018 13:50:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "data/slot_reader.h"
#include "data/info_parser.h"
#include "data/text_parser.h"
#include "util/recordio.h"
#include "util/threadpool.h"

namespace mltools {

void SlotReader::init(const DataConfig &data, const DataConfig &cache) {
  ASSERT_TRUE(cache.file_size() == 1);
  data_ = data;
  cache_ = cache.file(0);
}

std::string SlotReader::cacheName(const DataConfig &data, int slotId) const {
  CHECK_GE(data.file_size(), 1);
  return cache_ + data.file(0) + "_slot_" + std::to_string(slotId);
}

size_t SlotReader::nnzEle(int slotId) const {
  size_t nnz = 0;
  for (int i = 0; i < info_.slot_size(); ++i) {
    if (info_.slot(i).id() == slotId) {
      nnz = info_.slot(i).nnz_ele();
    }
  }

  return nnz;
}
} // namespace mltools
