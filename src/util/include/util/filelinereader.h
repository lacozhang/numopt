/*
 * =====================================================================================
 *
 *       Filename:  filelinereader.h
 *
 *    Description:  Read file line by line and invoke the call back function for
 * further processing
 *
 *        Version:  1.0
 *        Created:  07/11/2018 18:46:32
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/dataconfig.pb.h"
#include "util/file.h"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

namespace mltools {
class FileLineReader {
public:
  explicit FileLineReader(const DataConfig &data)
      : data_(data), loadSuccessfully_(false) {}
  ~FileLineReader() {}
  void setLineCallback(std::function<void(char *)> callback) {
    lineCallback_ = callback;
  }

  void reload();

  bool loadedSuccessfully() const { return loadSuccessfully_; }

private:
  DataConfig data_;
  bool loadSuccessfully_ = false;
  std::function<void(char *)> lineCallback_;
};
} // namespace mltools
