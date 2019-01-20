#pragma once
#include "IndexData.h"
#include "util/typedef.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

template <class SampleType, class LabelType> class DataIteratorBase {

public:
  typedef IndexData<SampleType, LabelType> IndexDataType;

  DataIteratorBase();
  ~DataIteratorBase() {}

  void InitFromCmd(int argc, const char *argv[]);
  boost::program_options::options_description Options() { return batchdesc_; }

  void SetDataSet(boost::shared_ptr<IndexDataType> dat);
  void ResetBatch();

  bool GetNextBatch(SampleType &batch, LabelType &label);
  bool GetRandomBatch(SampleType &batch, LabelType &label);
  bool IsValid() const { return valid_; }

  SampleType &GetAllData() const { return data_->RetrieveAllFeature(); }

  LabelType &GetAllLabel() const { return data_->RetrieveAllLabel(); }

  size_t GetSampleSize() const { return data_->SampleSize(); }

  size_t ModelSizeFromData() const {
    if (valid_) {
      return data_->ModelSize();
    } else {
      return -1;
    }
  }

private:
  void ConstructCmdOptions();

  // meta info
  int batchsize_;
  int seed_;
  bool valid_;

  // set iteration info
  size_t dataindex_;
  size_t realbatchsize_;
  std::vector<size_t> datapointers_;

  boost::shared_ptr<IndexDataType> data_;
  boost::program_options::options_description batchdesc_;

  static const char *kBaseBatchSizeOption;
  static const char *kBaseRandomSeedOption;
};
