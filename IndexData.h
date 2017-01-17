#pragma once
#ifndef __INDEX_DATA_H__
#define __INDEX_DATA_H__
#include<boost/shared_ptr.hpp>
#include "typedef.h"

template <class FeatureType, class LabelType>
class IndexData {
public:
	IndexData(boost::shared_ptr<FeatureType>& feature, boost::shared_ptr<LabelType>& label) {
		features_ = feature;
		labels_ = label;
	}
	virtual ~IndexData() {}

	inline void SetFeature(boost::shared_ptr<FeatureType>& feature) {
		features_ = feature;
	}

	inline void SetLabel(boost::shared_ptr<LabelType>& label) {
		labels_ = label;
	}

	size_t SampleSize() {
		BOOST_ASSERT_MSG(false, "Error, no specialization for this type");
		return 0;
	}

	bool FeatureCopyAtIndex(FeatureType& target, size_t topos, size_t frompos) {
		return false;
	}

	inline bool LabelCopyAtIndex(LabelType& target, size_t topos, size_t frompos) {
		return false;
	}

	inline FeatureType& RetrieveAllFeature() {
		return *features_;
	}

	inline LabelType& RetrieveAllLabel() {
		return *labels_;
	}

	void ResizeFeature(FeatureType& feature, size_t sample) {
		BOOST_ASSERT_MSG(false, "Error, no specialization for this type");
	}

	void ResizeLabel(LabelType& label, size_t sample) {
		BOOST_ASSERT_MSG(false, "Error, no specialization for this type");
	}

private:
	boost::shared_ptr<FeatureType> features_;
	boost::shared_ptr<LabelType> labels_;
};
#endif // !__INDEX_DATA_H__