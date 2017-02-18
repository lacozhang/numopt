#pragma once
#ifndef __INDEX_DATA_H__
#define __INDEX_DATA_H__
#include<boost/shared_ptr.hpp>
#include "typedef.h"
#include "LccrfDataType.h"

template <class FeatureType, class LabelType>
class IndexData {
public:
	IndexData(boost::shared_ptr<FeatureType>& feature, boost::shared_ptr<LabelType>& label) {

		if (feature.get() != nullptr && label.get() != nullptr) {
			valid_ = true;
			features_ = feature;
			labels_ = label;
		}
		else {
			valid_ = false;
		}
	}
	virtual ~IndexData() {}

	inline void SetFeature(boost::shared_ptr<FeatureType>& feature) {
		if (feature.get() != nullptr) {
			features_ = feature;
			valid_ = true;
		}
		else {
			valid_ = false;
		}
	}

	inline void SetLabel(boost::shared_ptr<LabelType>& label) {
		if (label.get() != nullptr) {
			labels_ = label;
			valid_ = true;
		}
		else {
			labels_.reset();
			valid_ = false;
		}
	}

	size_t SampleSize() {
		BOOST_ASSERT_MSG(false, "Error, no specialization for this type");
		return 0;
	}

	bool IsValid() {
		return valid_;
	}

	bool FeatureCopyAtIndex(FeatureType& target, size_t topos, size_t frompos) {
		BOOST_ASSERT_MSG(false, "Error, no specialization for this type");
		return false;
	}
	bool LabelCopyAtIndex(LabelType& target, size_t topos, size_t frompos) {
		BOOST_ASSERT_MSG(false, "Error, no specialization for this type");
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

	size_t ModelSize() {
		BOOST_ASSERT_MSG(false, "Error, no specialization for this type");
		return 0;
	}

private:
	boost::shared_ptr<FeatureType> features_;
	boost::shared_ptr<LabelType> labels_;
	bool valid_;
};

// specialization for liblinear type data
template<> 
void IndexData<DataSamples, LabelVector>::ResizeFeature(DataSamples& feature, size_t sample);

template<>
void IndexData<DataSamples, LabelVector>::ResizeLabel(LabelVector& label, size_t sample);

template<>
bool IndexData<DataSamples, LabelVector>::FeatureCopyAtIndex(DataSamples& target, size_t topos, size_t frompos);

template<>
bool IndexData<DataSamples, LabelVector>::LabelCopyAtIndex(LabelVector& target, size_t topos, size_t frompos);

template<>
size_t IndexData<DataSamples, LabelVector>::SampleSize();


// specialization for lccrf type data
template<>
void IndexData<LccrfSamples, LccrfLabels>::ResizeFeature(LccrfSamples& feature, size_t sample);

template<>
void IndexData<LccrfSamples, LccrfLabels>::ResizeLabel(LccrfLabels& label, size_t sample);

template<>
bool IndexData<LccrfSamples, LccrfLabels>::FeatureCopyAtIndex(LccrfSamples& target, size_t topos, size_t frompos);

template<>
bool IndexData<LccrfSamples, LccrfLabels>::LabelCopyAtIndex(LccrfLabels& target, size_t topos, size_t frompos);

template<>
size_t IndexData<LccrfSamples, LccrfLabels>::SampleSize();

template<>
size_t IndexData<LccrfSamples, LccrfLabels>::ModelSize();
#endif // !__INDEX_DATA_H__