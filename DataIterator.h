#pragma once
#include <vector>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include "IndexData.h"
#include "typedef.h"

template <class SampleType, class LabelType>
class DataIteratorBase {

public:

	typedef IndexData<SampleType, LabelType> IndexDataType;

	DataIteratorBase();
	~DataIteratorBase() {}

	void InitFromCmd(int argc, const char* argv[]);
	boost::program_options::options_description Options() {
		return batchdesc_;
	}

	void SetDataSet(boost::shared_ptr<IndexDataType>& dat);	
	void ResetBatch();

	bool GetNextBatch(SampleType& batch, LabelType& label);
	bool IsValid() const {
		return valid_;
	}

	SampleType& GetAllData() const {
		return data_->RetrieveAllFeature();
	}

	LabelType& GetAllLabel() const {
		return data_->RetrieveAllLabel();
	}

	size_t GetSampleSize() const {
		return data_->SampleSize();
	}

	size_t MaxFeatureId() const {
		if (valid_) {
			return maxfeatid_;
		}
		else {
			return -1;
		}
	}
	
private:

	void ConstructCmdOptions();

	// meta info
	int batchsize_;
	int seed_;
	bool valid_;
	size_t maxfeatid_;

	// set iteration info
	size_t dataindex_;
	size_t realbatchsize_;
	std::vector<size_t> datapointers_;

	boost::shared_ptr<IndexDataType> data_;
	boost::program_options::options_description batchdesc_;

	static const char* kBaseBatchSizeOption;
	static const char* kBaseRandomSeedOption;
};