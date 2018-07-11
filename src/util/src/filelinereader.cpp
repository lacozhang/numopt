/*
 * =====================================================================================
 *
 *       Filename:  filelinereader.cpp
 *
 *    Description:  implement the interface of filelinereader.h
 *
 *        Version:  1.0
 *        Created:  07/11/2018 18:47:06
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "util/filelinereader.h"
#include "util/common.h"
#include "util/file.h"
#include <cstring>
#include <glog/logging.h>
#include <memory>

namespace {
constexpr int kMaxLineLength = 60 * 1024;
}

namespace mltools {

DEFINE_int32(line_limit, 0, "maximum number of lines can read from one file");

void FileLineReader::reload() {
  File *f = File::open(data_, "r");
  if (f == NULL || !f->isOpen()) {
    loadSuccessfully_ = false;
    LOG(ERROR) << "Failed to open file " << data_.DebugString();
    return;
  }

  std::unique_ptr<char[]> buffer(new char[kMaxLineLength]);
  size_t lineCounts = 0;
  if (!buffer.get()) {
    LOG(FATAL) << "Failed to allocate " << kMaxLineLength
               << " bytes for read buffer";
    return;
  }

  loadSuccessfully_ = true;
  auto succ = f->readLine(buffer.get(), kMaxLineLength);
  while (succ != nullptr) {
    ++lineCounts;
    size_t lineLen = std::strlen(succ);
    while (lineLen > 0 &&
           (succ[lineLen - 1] == '\r' || succ[lineLen - 1] == '\n')) {
      succ[--lineLen] = '\0';
    }
    if (lineCallback_ && lineLen > 0) {
      lineCallback_(buffer.get());
    }
    if (FLAGS_line_limit > 0 && lineCounts >= FLAGS_line_limit) {
      break;
    }
  }
}
} // namespace mltools
