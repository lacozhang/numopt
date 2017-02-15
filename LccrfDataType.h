#pragma once

#ifndef __LCCRF_DATA_TYPE_H__
#define __LCCRF_DATA_TYPE_H__
#include <vector>
#include <boost/shared_ptr.hpp>
#include "typedef.h"

class LccrfSamples {
public:
	LccrfSamples() {
		maxunifeatid_ = maxbifeatid_ = 0;
		unigramfeatures_.clear();
		bigramfeatures_.clear();
	}
	
	std::vector<boost::shared_ptr<DataSamples>>& UnigramFeature() {
		return unigramfeatures_;
	}

	std::vector<boost::shared_ptr<DataSamples>>& BigramFeature() {
		return bigramfeatures_;
	}

	boost::shared_ptr<DataSamples>& UnigramFeature(size_t idx) {
		return unigramfeatures_[idx];
	}

	boost::shared_ptr<DataSamples>& BigramFeature(size_t idx) {
		return bigramfeatures_[idx];
	}

	size_t GetMaxUnigramFeatureId() {
		return maxunifeatid_;
	}

	void SetMaxUnigramFeatureId(size_t val) {
		maxunifeatid_ = val;
	}

	size_t GetMaxBigramFeatureId() {
		return maxbifeatid_;
	}

	void SetMaxBigramFeatureId(size_t val) {
		maxbifeatid_ = val;
	}

	size_t NumSamples() const {
		return unigramfeatures_.size();
	}

private:
	std::vector<boost::shared_ptr<DataSamples>> unigramfeatures_, bigramfeatures_;
	size_t maxunifeatid_, maxbifeatid_;
};

class LccrfLabels {
public:
	LccrfLabels() {
		labels_.clear();
	}
	std::vector<boost::shared_ptr<LabelVector>>& Labels() {
		return labels_;
	}

	boost::shared_ptr<LabelVector>& Labels(size_t idx) {
		return labels_[idx];
	}

	size_t GetMaxLabelId() {
		return maxlabelid_;
	}

	size_t NumSamples() const {
		return labels_.size();
	}

	void SetMaxLabelId(size_t val) {
		maxlabelid_ = val;
	}

private:
	std::vector<boost::shared_ptr<LabelVector>> labels_;
	size_t maxlabelid_;
};

#endif // !__LCCRF_DATA_TYPE_H__
