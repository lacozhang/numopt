#include "IndexData.h"


// definition for Classification datas
template<>
size_t IndexData<DataSamples, LabelVector>::SampleSize() {
	return features_->rows();
}

template<>
bool IndexData<DataSamples, LabelVector>::FeatureCopyAtIndex(DataSamples& target, size_t topos, size_t frompos) {
	if ((frompos < features_->rows()) && (topos < target.rows())) {
		target.row(topos) = features_->row(frompos);
		return true;
	}
	else {
		return false;
	}
}

template<>
bool IndexData<DataSamples, LabelVector>::LabelCopyAtIndex(LabelVector& target, size_t topos, size_t frompos) {
	if ((frompos < labels_->rows()) && (topos < target.rows())) {
		target.row(topos) = labels_->row(frompos);
		return true;
	}
	else {
		return false;
	}
}

template<>
void IndexData<DataSamples, LabelVector>::ResizeFeature(DataSamples& target, size_t samplecnt) {
	target.resize(samplecnt, features_->cols());
}

template<>
void IndexData<DataSamples, LabelVector>::ResizeLabel(LabelVector& target, size_t samplecnt) {
	target.resize(samplecnt);
}


template class IndexData<DataSamples, LabelVector>;