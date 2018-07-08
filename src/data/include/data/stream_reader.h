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

#include "util/dynamic_array_impl.h"
#include "proto/example.pb.h"
#include "proto/matrix.pb.h"
#include "data/text_parser.h"
#include "data/info_parser.h"
#include "util/sparse_matrix.h"
#include "data/common.h"

namespace mltools {
  DECLARE_uint64(hash_kernel);
  
  template <typename V>
  class StreamReader {
  public:
    StreamReader() {}
    StreamReader(const DataConfig &conf) {
      init(conf);
    }
    
    void init(const DataConfig &conf);
    
    /**
     * @brief Read #numExamples examples from data
     *
     * Read "numExamples" examples from training data in matrix format, if error happens or reach end of files return false; otherwise, return true
     *
     */
    bool readMatrices(uint32 numExamples, MatrixPtrList<V> *matrices, std::vector<Example> *examples = nullptr);
    
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
      bool empty() {
        return cnt_.empty() && idx_.empty() && val_.empty();
      }
    };
    
    std::vector<VSlot> vslots_;
    ExampleParser textParser_;
    InfoParser infoParser_;
    DataConfig data_;
    
    int nextFileIdx_ = 0;
    int maxNumFiles_ = 0;
    static constexpr int kMaxLineLength_ = 60*1024;
    char line_[kMaxLineLength_];
    File *dataFile_ = nullptr;
    bool reachDataEnd_ = false;
    
    // input from caller
    uint32 numExamples_ = 0;
    MatrixPtrList<V> *matrices_ = nullptr;
    std::vector<Example> *examples_ = nullptr;
  };
  
  template <typename V>
  bool StreamReader<V>::openNextFile() {
    if(dataFile_ != nullptr) {
      dataFile_->close();
      dataFile_ = nullptr;
    }
    while(nextFileIdx_ < maxNumFiles_ && !reachDataEnd_) {
      dataFile_ = File::open(data_.file(nextFileIdx_++), "r");
      if(dataFile_ != nullptr) {
        break;
      }
    }
    
    if(nextFileIdx_ >= maxNumFiles_) {
      reachDataEnd_ = true;
    }
    return dataFile_ != nullptr;
  }
  
  template <typename V>
  void StreamReader<V>::init(const DataConfig &conf) {
    data_ = conf;
    nextFileIdx_ = 0;
    maxNumFiles_ = conf.file_size();
    if(conf.maximum_files_per_worker() > 0) {
      maxNumFiles_ = std::min(maxNumFiles_, conf.maximum_files_per_worker());
    }
    if(conf.format() == DataConfig::TEXT) {
      textParser_.init(conf.text(), conf.ignore_feature_group());
    }
    reachDataEnd_ = false;
    openNextFile();
  }
  
  template <typename V>
  void StreamReader<V>::parseExample(const Example &ex, int numReads) {
    if(!infoParser_.add(ex)) {
      return;
    }
    for(int i=0; i<ex.slot_size(); ++i) {
      auto& slot = ex.slot(i);
      auto& vslot = vslots_[slot.id()];
      vslot.val_.ad
    }
  }
  
  
}

#endif // __STREAM_READER_H__
