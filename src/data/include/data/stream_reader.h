/*
 * =====================================================================================
 *
 *       Filename:  stream_reader.h
 *
 *    Description:  read training data in stream format
 *
 *        Version:  1.0
 *        Created:  07/08/2018 13:48:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#ifndef __STREAM_READER_H__
#define __STREAM_READER_H__

#include "data/common.h"
#include "data/info_parser.h"
#include "data/text_parser.h"
#include "proto/example.pb.h"
#include "proto/matrix.pb.h"
#include "util/dynamic_array_impl.h"
#include "util/recordio.h"
#include "util/sparse_matrix.h"

namespace mltools {
DECLARE_uint64(hash_kernel);

template <typename V> class StreamReader {
public:
  StreamReader() {}
  StreamReader(const DataConfig &conf) { init(conf); }

  void init(const DataConfig &conf);

  /**
   * @brief Read #numExamples examples from data
   *
   * Read "numExamples" examples from training data in matrix format, if error
   * happens or reach end of files return false; otherwise, return true
   *
   */
  bool readMatrices(uint32 numExamples, MatrixPtrList<V> *matrices,
                    std::vector<Example> *examples = nullptr);

  /// @brief read training data in Example format
  bool readExamples(uint32 numExamples, std::vector<Example> *examples) {
    return readMatrices(numExamples, nullptr, examples);
  }

private:
  bool readMatricesFromText();
  bool readMatricesFromProto();
  void parseExample(const Example &ex, int numReads);
  void fillMatrices();

  bool openNextFile();
  struct VSlot {
    DArray<V> val_;
    DArray<uint64> idx_;
    DArray<uint16> cnt_;
    bool empty() { return cnt_.empty() && idx_.empty() && val_.empty(); }
  };

  std::vector<VSlot> vslots_;
  ExampleParser textParser_;
  InfoParser infoParser_;
  DataConfig data_;

  int nextFileIdx_ = 0;
  int maxNumFiles_ = 0;
  static constexpr int kMaxLineLength_ = 60 * 1024;
  char line_[kMaxLineLength_];
  File *dataFile_ = nullptr;
  bool reachDataEnd_ = false;

  // input from caller
  uint32 numExamples_ = 0;
  MatrixPtrList<V> *matrices_ = nullptr;
  std::vector<Example> *examples_ = nullptr;
};

template <typename V> bool StreamReader<V>::openNextFile() {
  if (dataFile_ != nullptr) {
    dataFile_->close();
    dataFile_ = nullptr;
  }
  while (true) {
    if (nextFileIdx_ >= maxNumFiles_) {
      reachDataEnd_ = true;
      return false;
    }
    dataFile_ = File::open(data_.file(nextFileIdx_++), "r");
    if (dataFile_ != nullptr) {
      break;
    }
  }
  return true;
}

template <typename V> void StreamReader<V>::init(const DataConfig &conf) {
  data_ = conf;
  nextFileIdx_ = 0;
  maxNumFiles_ = conf.file_size();
  if (conf.maximum_files_per_worker() > 0) {
    maxNumFiles_ = std::min(maxNumFiles_, conf.maximum_files_per_worker());
  }
  if (conf.format() == DataConfig::TEXT) {
    textParser_.init(conf.text(), conf.ignore_feature_group());
  }
  reachDataEnd_ = false;
  openNextFile();
}

template <typename V>
void StreamReader<V>::parseExample(const Example &ex, int numReads) {
  if (examples_) {
    examples_->push_back(ex);
  }
  if (!matrices_) {
    return;
  }
  if (!infoParser_.add(ex)) {
    return;
  }
  for (int i = 0; i < ex.slot_size(); ++i) {
    auto &slot = ex.slot(i);
    CHECK_LT(slot.id(), static_cast<size_t>(FeatureConstants::kSlotIdMax));
    auto &vslot = vslots_[slot.id()];
    auto keySize = slot.key_size();
    if (FLAGS_hash_kernel > 0) {
      for (int i = 0; i < keySize; ++i) {
        vslot.idx_.push_back(slot.key(i) % FLAGS_hash_kernel);
      }
    } else {
      for (int i = 0; i < keySize; ++i) {
        vslot.idx_.push_back(slot.key(i));
      }
    }

    auto valSize = slot.val_size();
    for (int i = 0; i < valSize; ++i) {
      vslot.val_.push_back(slot.val(i));
    }

    while (vslot.cnt_.size() < numReads) {
      vslot.cnt_.push_back(0);
    }

    vslot.cnt_.push_back(std::max(keySize, valSize));
  }
}

template <typename V> bool StreamReader<V>::readMatricesFromText() {
  int numReads = 0;
  while (numReads < numExamples_ && !reachDataEnd_) {
    while (true) {
      auto ret = dataFile_->readLine(line_, kMaxLineLength_);
      if (ret != nullptr) {
        int len = std::strlen(ret);
        while (len > 0 && (ret[len - 1] == '\n' || ret[len - 1] == '\r')) {
          --len;
        }
        Example ex;
        if (textParser_.toProto(line_, &ex)) {
          parseExample(ex, numReads);
          numReads++;
          break;
        }
      } else {
        if (!openNextFile()) {
          break;
        }
      }
    }
  }
  fillMatrices();
  return !reachDataEnd_;
}

template <typename V> bool StreamReader<V>::readMatricesFromProto() {
  int numReads = 0;
  auto reader(dataFile_);
  Example ex;
  while (numReads < numExamples_ && !reachDataEnd_) {
    while (true) {
      if (reader.readProtoMessage(&ex)) {
        parseExample(&ex, numReads);
        ++numReads;
        break;
      } else {
        if (!openNextFile()) {
          break;
        }
        reader = RecordReader(dataFile_);
      }
    }
  }

  fillMatrices();
  return !reachDataEnd_;
}

template <typename V>
bool StreamReader<V>::readMatrices(uint32 numExamples,
                                   MatrixPtrList<V> *matrices,
                                   std::vector<Example> *examples) {
  if (numExamples <= 0) {
    return true;
  }
  numExamples_ = numExamples;
  matrices_ = matrices;
  if (matrices_) {
    matrices_->clear();
  }
  examples_ = examples;
  if (examples_) {
    examples_->clear();
  }

  vslots_.resize(data_.ignore_feature_group()
                     ? 2
                     : static_cast<int>(FeatureConstants::kSlotIdMax));
  bool ret = false;
  if (data_.format() == DataConfig::TEXT) {
    ret = readMatricesFromText();
  } else if (data_.format() == DataConfig::BIN) {
    ret = readMatricesFromProto();
  } else {
    LOG(ERROR) << "Unknow data format " << data_.DebugString();
  }
  vslots_.clear();
  return ret;
}

template <typename V> void StreamReader<V>::fillMatrices() {
  if (!matrices_) {
    return;
  }

  auto info = infoParser_.info();
  infoParser_.clear();
  for (int i = 0; i < static_cast<int>(FeatureConstants::kSlotIdMax); ++i) {
    auto &slot = vslots_[i];
    if (slot.empty()) {
      continue;
    }
    auto minfo = readMatrixInfo(info, i, sizeof(uint64), sizeof(V));
    if (minfo.type() == MatrixInfo::DENSE) {
      matrices_->emplace_back(
          MatrixPtr<V>(new DenseMatrix<V>(minfo, slot.val_)));
    } else {
      int rs = slot.cnt_.size();
      DArray<size_t> offset(rs + 1);
      offset[0] = 0;
      for (int i = 0; i < rs; ++i) {
        offset[i + 1] = offset[i] + slot.cnt_[i];
      }

      matrices_->emplace_back(MatrixPtr<V>(
          new SparseMatrix<uint64, V>(minfo, offset, slot.idx_, slot.val_)));

      CHECK_EQ(offset.back(), slot.idx_.size());
      if (!slot.val_.empty()) {
        CHECK_EQ(offset.back(), slot.val_.size());
      }
    }
  }
}

} // namespace mltools

#endif // __STREAM_READER_H__
