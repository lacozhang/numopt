#pragma once

#ifndef __NNSEQUENCE_DATA_H__
#define __NNSEQUENCE_DATA_H__
#include "util/typedef.h"
#include <boost/shared_ptr.hpp>
#include <vector>
#include <glog/logging.h>

namespace NNModel {

class SentenceFeature {
public:
  SentenceFeature() {}

  ~SentenceFeature() {}

  DataSamples &SparseBinaryFeature() { return sparsebinary_; }

  DataSamples::ConstRowXpr SparseBinaryFeature(int pos) const {
    return sparsebinary_.row(pos);
  }

  DataSamples &SparseFeature() { return sparse_; }

  DataSamples::ConstRowXpr SparseFeature(int pos) const {
    return sparse_.row(pos);
  }

  DenseMatrix &DenseFeature() { return dense_; }

  DenseMatrix::ConstRowXpr DenseFeature(int pos) const {
    return dense_.row(pos);
  }

private:
  DataSamples sparsebinary_;
  DataSamples sparse_;
  DenseMatrix dense_;
};

class SentenceLabel {
public:
  const static int UNKNOWN = -1;

  SentenceLabel() { labels_.resize(0); }

  explicit SentenceLabel(int size) {
    labels_.resize(size);
    for (int i = 0; i < labels_.size(); ++i) {
      labels_.coeffRef(i) = SentenceLabel::UNKNOWN;
    }
  }

  ~SentenceLabel() {}

  int GetLabel(int idx) { return labels_[idx]; }

  void SetLabel(int idx, int label) { labels_[idx] = label; }

  LabelVector &GetLabels() { return labels_; }

  void SetLabels(std::vector<int> &labels) {
    labels_.resize(labels.size());
    for (int i = 0; i < labels.size(); ++i)
      labels_.coeffRef(i) = labels[i];
  }

  void SetLabels(LabelVector &labels) { labels_ = labels; }

private:
  LabelVector labels_;
};

class NNSequenceFeature {
public:
  NNSequenceFeature() {
    features_.clear();
    spbinarysize_ = spfloatsize_ = densesize_ = 0;
    nullfeat_.reset();
  }
  ~NNSequenceFeature() {}

  void AppendSequenceFeature(boost::shared_ptr<SentenceFeature> &feat) {
    features_.push_back(std::move(feat));
  }

  void SetSequenceFeature(boost::shared_ptr<SentenceFeature> &feat, int index) {
    while (index >= features_.size()) {
      LOG(INFO) << "increase the size of features";
      features_.resize(2 * features_.size());
    }

    features_[index] = feat;
  }

  boost::shared_ptr<SentenceFeature> &GetSequenceFeature(int index) {
    if (index < features_.size()) {
      return features_[index];
    }
    LOG(INFO) << "Access data index outside of array";
    return nullfeat_;
  }

  std::vector<boost::shared_ptr<SentenceFeature>> &SampleFeatures() {
    return features_;
  }

  boost::shared_ptr<SentenceFeature> &operator[](int index) {
    return features_[index];
  }

  size_t NumSamples() { return features_.size(); }

  size_t GetSparseBinarySize() { return spbinarysize_; }

  void SetSparseBinarySize(size_t size) { spbinarysize_ = size; }

  size_t GetSparseFloatSize() { return spfloatsize_; }

  void SetSparseFloatSize(size_t size) { spfloatsize_ = size; }

  size_t GetDenseSize() { return densesize_; }

  void SetDenseSize(size_t size) { densesize_ = size; }

private:
  std::vector<boost::shared_ptr<SentenceFeature>> features_;
  boost::shared_ptr<SentenceFeature> nullfeat_;
  size_t spbinarysize_, spfloatsize_, densesize_;
};

class NNSequenceLabel {
public:
  NNSequenceLabel() {
    labels_.clear();
    labelsize_ = 0;
    nullabel_.reset();
  }
  ~NNSequenceLabel() {}

  void AppendSequenceLabel(boost::shared_ptr<SentenceLabel> &label) {
    labels_.push_back(std::move(label));
  }

  void SetSequenceLabel(boost::shared_ptr<SentenceLabel> &label, int idx) {
    while (idx >= labels_.size()) {
      LOG(INFO) << "Increase the size of labels";
      labels_.resize(2 * labels_.size());
    }

    labels_[idx] = label;
  }

  boost::shared_ptr<SentenceLabel> &GetSequenceLabel(int index) {
    if (index < labels_.size())
      return labels_[index];

    LOG(INFO) << "Access data index outside of array";
    return nullabel_;
  }

  std::vector<boost::shared_ptr<SentenceLabel>> &SampleLabels() {
    return labels_;
  }
  boost::shared_ptr<SentenceLabel> &operator[](int index) {
    return labels_[index];
  }

  size_t GetLabelSize() { return labelsize_; }

  void SetLabelSize(size_t numlabels) { labelsize_ = numlabels; }

  size_t NumSamples() { return labels_.size(); }

private:
  std::vector<boost::shared_ptr<SentenceLabel>> labels_;
  boost::shared_ptr<SentenceLabel> nullabel_;
  size_t labelsize_;
};

struct NNSequenceEstimate {
  int sparsebinarysize_, sparsefloatsize_, densesize_, labelsize_;
  NNSequenceEstimate() {
    sparsebinarysize_ = sparsefloatsize_ = densesize_ = labelsize_ = 0;
  }
};

struct NNSequenceParse {
  std::vector<boost::shared_ptr<SentenceFeature>> feats_;
  std::vector<boost::shared_ptr<SentenceLabel>> labels_;

  NNSequenceParse() {
    feats_.clear();
    labels_.clear();
  }
};
} // namespace NNModel

#endif // !__NNSEQUENCE_DATA_H__
