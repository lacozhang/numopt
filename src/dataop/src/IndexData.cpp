#include "dataop/IndexData.h"

// definition for Classification datas
template <> size_t IndexData<DataSamples, LabelVector>::SampleSize() {
  return features_->rows();
}

template <> size_t IndexData<LccrfSamples, LccrfLabels>::SampleSize() {
  return features_->NumSamples();
}

template <>
bool IndexData<DataSamples, LabelVector>::FeatureCopyAtIndex(
    DataSamples &target, size_t topos, size_t frompos) {
  if ((frompos < features_->rows()) && (topos < target.rows())) {
    target.row(topos) = features_->row(frompos);
    return true;
  } else {
    return false;
  }
}

template <>
bool IndexData<LccrfSamples, LccrfLabels>::FeatureCopyAtIndex(
    LccrfSamples &target, size_t topos, size_t frompos) {
  if ((frompos) < features_->NumSamples() && (topos < target.NumSamples())) {
    target.UnigramFeature(topos) = features_->UnigramFeature(frompos);
    target.BigramFeature(topos) = features_->BigramFeature(frompos);
    return true;
  } else {
    return false;
  }
}

template <>
bool IndexData<DataSamples, LabelVector>::LabelCopyAtIndex(LabelVector &target,
                                                           size_t topos,
                                                           size_t frompos) {
  if ((frompos < labels_->rows()) && (topos < target.rows())) {
    target.row(topos) = labels_->row(frompos);
    return true;
  } else {
    return false;
  }
}

template <>
bool IndexData<LccrfSamples, LccrfLabels>::LabelCopyAtIndex(LccrfLabels &target,
                                                            size_t topos,
                                                            size_t frompos) {
  if ((frompos) < labels_->NumSamples() && (topos < target.NumSamples())) {
    target.Labels(topos) = labels_->Labels(frompos);
    return true;
  } else {
    return false;
  }
}

template <>
void IndexData<DataSamples, LabelVector>::ResizeFeature(DataSamples &target,
                                                        size_t samplecnt) {
  target.resize(samplecnt, features_->cols());
}

template <>
void IndexData<LccrfSamples, LccrfLabels>::ResizeFeature(LccrfSamples &target,
                                                         size_t samplecnt) {
  target.UnigramFeature().resize(samplecnt);
  target.BigramFeature().resize(samplecnt);
  target.SetMaxUnigramFeatureId(features_->GetMaxUnigramFeatureId());
  target.SetMaxBigramFeatureId(features_->GetMaxBigramFeatureId());
}

template <>
void IndexData<DataSamples, LabelVector>::ResizeLabel(LabelVector &target,
                                                      size_t samplecnt) {
  target.resize(samplecnt);
}

template <>
void IndexData<LccrfSamples, LccrfLabels>::ResizeLabel(LccrfLabels &target,
                                                       size_t samplecnt) {
  target.Labels().resize(samplecnt);
  target.SetMaxLabelId(labels_->GetMaxLabelId());
}

template <> size_t IndexData<LccrfSamples, LccrfLabels>::ModelSize() {
  size_t lccrfsize = 0;
  size_t labelsize = labels_->GetMaxLabelId() + 1;
  lccrfsize += (features_->GetMaxUnigramFeatureId() + 1) * labelsize;
  lccrfsize += (features_->GetMaxBigramFeatureId() + 1) * labelsize * labelsize;
  return lccrfsize;
}

// specialization for nn classification data type
template <>
void IndexData<NNModel::NNQueryFeature, NNModel::NNQueryLabel>::ResizeFeature(
    NNModel::NNQueryFeature &feature, size_t numsample) {
  feature.Features().resize(numsample);
}
template <>
void IndexData<NNModel::NNQueryFeature, NNModel::NNQueryLabel>::ResizeLabel(
    NNModel::NNQueryLabel &label, size_t numsample) {
  label.Labels().resize(numsample);
}

template <>
bool IndexData<NNModel::NNQueryFeature, NNModel::NNQueryLabel>::
    FeatureCopyAtIndex(NNModel::NNQueryFeature &target, size_t topos,
                       size_t frompos) {
  if (topos < target.NumSamples() && frompos < features_->NumSamples()) {
    target.FeatureOfSample(topos) = features_->FeatureOfSample(frompos);
    return true;
  } else
    return false;
}

template <>
bool IndexData<NNModel::NNQueryFeature, NNModel::NNQueryLabel>::
    LabelCopyAtIndex(NNModel::NNQueryLabel &target, size_t topos,
                     size_t frompos) {
  if (topos < target.NumSamples() && frompos < labels_->NumSamples()) {
    target.LabelOfSample(topos) = labels_->LabelOfSample(frompos);
    return true;
  } else
    return false;
}
template <>
size_t IndexData<NNModel::NNQueryFeature, NNModel::NNQueryLabel>::SampleSize() {
  if (features_->NumSamples() != labels_->NumSamples()) {
    LOG(ERROR) << "Samples of feature do not equal samples of labels";
  }
  return features_->NumSamples();
}
template <>
size_t IndexData<NNModel::NNQueryFeature, NNModel::NNQueryLabel>::ModelSize() {
  LOG(INFO) << "do not use this to estimate model size";
  return 0;
}

// specialization for nn sequence labelling type
template <>
void IndexData<NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::
    ResizeFeature(NNModel::NNSequenceFeature &feature, size_t sample) {
  feature.SampleFeatures().resize(sample);
}
template <>
void IndexData<NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::
    ResizeLabel(NNModel::NNSequenceLabel &label, size_t sample) {
  label.SampleLabels().resize(sample);
}
template <>
bool IndexData<NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::
    FeatureCopyAtIndex(NNModel::NNSequenceFeature &target, size_t topos,
                       size_t frompos) {
  if (topos < target.NumSamples() && frompos < features_->NumSamples()) {
    target.GetSequenceFeature(topos) = features_->GetSequenceFeature(frompos);
    return true;
  } else
    return false;
}

template <>
bool IndexData<NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::
    LabelCopyAtIndex(NNModel::NNSequenceLabel &target, size_t topos,
                     size_t frompos) {
  if (topos < target.NumSamples() && frompos < labels_->NumSamples()) {
    target.GetSequenceLabel(topos) = labels_->GetSequenceLabel(frompos);
    return true;
  } else
    return false;
}

template <>
size_t
IndexData<NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::SampleSize() {
  return features_->NumSamples();
}

template <>
size_t
IndexData<NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::ModelSize() {
  LOG(INFO) << "do not use this to estimate model size";
  return 0;
}

template class IndexData<DataSamples, LabelVector>;
template class IndexData<LccrfSamples, LccrfLabels>;
template class IndexData<NNModel::NNQueryFeature, NNModel::NNQueryLabel>;
template class IndexData<NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>;
