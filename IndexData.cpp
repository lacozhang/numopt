#include "IndexData.h"


// definition for Classification datas
template<>
size_t IndexData<DataSamples, LabelVector>::SampleSize() {
	return features_->rows();
}

template<>
size_t IndexData<LccrfSamples, LccrfLabels>::SampleSize() {
	return features_->NumSamples();
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
bool IndexData<LccrfSamples, LccrfLabels>::FeatureCopyAtIndex(LccrfSamples& target, size_t topos, size_t frompos) {
	if ((frompos) < features_->NumSamples() && (topos < target.NumSamples())) {
		target.UnigramFeature(topos) = features_->UnigramFeature(frompos);
		target.BigramFeature(topos) = features_->BigramFeature(frompos);
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
bool IndexData<LccrfSamples, LccrfLabels>::LabelCopyAtIndex(LccrfLabels& target, size_t topos, size_t frompos) {
	if ((frompos) < labels_->NumSamples() && (topos < target.NumSamples())) {
		target.Labels(topos) = labels_->Labels(frompos);
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
void IndexData<LccrfSamples, LccrfLabels>::ResizeFeature(LccrfSamples& target, size_t samplecnt) {
	target.UnigramFeature().resize(samplecnt);
	target.BigramFeature().resize(samplecnt);
}

template<>
void IndexData<DataSamples, LabelVector>::ResizeLabel(LabelVector& target, size_t samplecnt) {
	target.resize(samplecnt);
}

template<>
void IndexData<LccrfSamples, LccrfLabels>::ResizeLabel(LccrfLabels& target, size_t samplecnt) {
	target.Labels().resize(samplecnt);
}

template<>
size_t IndexData<DataSamples, LabelVector>::FeatureSize() {
	return features_->cols();
}

template<>
size_t IndexData<LccrfSamples, LccrfLabels>::FeatureSize() {
	return 0;
}

template<>
size_t IndexData<DataSamples, LabelVector>::MaxFeatureId() {
	return (features_->cols() - 1);
}

template<>
size_t IndexData<LccrfSamples, LccrfLabels>::ModelSize() {
	size_t lccrfsize = 0;
	size_t labelsize = labels_->GetMaxLabelId() + 1;
	lccrfsize += (features_->GetMaxUnigramFeatureId() + 1)*labelsize;
	lccrfsize += (features_->GetMaxBigramFeatureId() + 1)*labelsize*labelsize;
	return lccrfsize;
}

template class IndexData<DataSamples, LabelVector>;
template class IndexData<LccrfSamples, LccrfLabels>;