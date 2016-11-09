#pragma once
#include <vector>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <boost/log/trivial.hpp>
#include "typedef.h"

class DataIterator {

public:
	DataIterator(int batchsize, int seed);
	~DataIterator() {}

	void SetDataSet(boost::shared_ptr<DataSamples>& data, boost::shared_ptr<LabelVector>& label);
	
	void ResetBatch();
	bool GetNextBatch(DataSamples& batch, LabelVector& label);
	bool IsValid() const {
		return valid_;
	}

	DataSamples& GetAllData() const {
		return *dataset_;
	}

	LabelVector& GetAllLabel() const {
		return *labels_;
	}
	
private:
	// meta info
	int batchsize_;
	int seed_;
	bool valid_;

	// set iteration info
	size_t dataindex_;
	size_t realbatchsize_;
	std::vector<size_t> datapointers_;

	boost::shared_ptr<DataSamples> dataset_;
	boost::shared_ptr<LabelVector> labels_;
};