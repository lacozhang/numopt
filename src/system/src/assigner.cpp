/*
 * =====================================================================================
 *
 *       Filename:  assigner.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:23:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "system/assigner.h"
#include "data/common.h"

namespace mltools {
void DataAssigner::set(const mltools::DataConfig &data, int num, bool local) {
  CHECK_GT(num, 0);
  parts_.resize(num);
  if (local) {
    for (int i = 0; i < num; ++i) {
      parts_[i].CopyFrom(data);
    }
    return;
  }

  CHECK_GT(data.replica(), 0);
  DataConfig files = searchFiles(data);
  VLOG(1) << "find " << files.file_size() << "files"
          << "files : " << files.ShortDebugString();

  for (int r = 0; r < data.replica(); ++r) {
    if (data.shuffle()) {
      shuffleFiles(files);
    }
    auto prts = divideFiles(files, num);
    if (r == 0) {
      parts_ = prts;
    } else {
      for (int i = 0; i < num; ++i) {
        parts_[i] = prts[i];
      }
    }
  }
  VLOG(1) << "divide files into " << num << " parts";
}

bool DataAssigner::next(mltools::DataConfig *data) {
  if (currIdx_ >= parts_.size()) {
    return false;
  }
  data->CopyFrom(parts_[currIdx_]);
  ++currIdx_;
  return true;
}
} // namespace mltools
