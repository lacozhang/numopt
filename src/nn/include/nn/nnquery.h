#pragma once

#include "dataop/vocabulary.h"
#include "util/stringop.h"
#include "util/typedef.h"
#include <boost/algorithm/string.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/signals2/detail/auto_buffer.hpp>
#include <glog/logging.h>

#ifndef __NN_QUERY_H__
#define __NN_QUERY_H__

namespace NNModel {
class QueryFeature {
public:
  QueryFeature() {}
  ~QueryFeature() {}

  DataSamples &Feature() { return features_; }

private:
  DataSamples features_;
};

class QueryLabel {
public:
  QueryLabel() {}
  ~QueryLabel() {}

  int &Label() { return labels_; }

private:
  int labels_;
};

class NNQueryFeature {
public:
  NNQueryFeature() { featdat_.clear(); }
  ~NNQueryFeature() {}

  std::vector<boost::shared_ptr<QueryFeature>> &Features() { return featdat_; }

  void SetVocabulary(boost::shared_ptr<Vocabulary> &vocab) { vocab_ = vocab; }

  boost::shared_ptr<Vocabulary> &GetVocabulary() { return vocab_; }

  void AppendQueryFeature(boost::shared_ptr<QueryFeature> &feat) {
    featdat_.push_back(feat);
  }

  void InsertQueryFeature(int idx, boost::shared_ptr<QueryFeature> &feat) {
    while (idx >= featdat_.size()) {
      featdat_.resize(featdat_.size() * 2);
      LOG(INFO) << "Expand feature data size";
    }

    featdat_[idx] = feat;
  }

  size_t GetVocabularySize() { return vocab_->VocabSize(); }

  boost::shared_ptr<QueryFeature> &FeatureOfSample(int idx) {
    if (idx >= featdat_.size()) {
      LOG(FATAL) << "Access Sample Index out of bound";
      std::abort();
    }
    return featdat_[idx];
  }

  size_t NumSamples() { return featdat_.size(); }

private:
  std::vector<boost::shared_ptr<QueryFeature>> featdat_;
  boost::shared_ptr<Vocabulary> vocab_;
};

class NNQueryLabel {
public:
  NNQueryLabel() { labels_.clear(); }
  ~NNQueryLabel() {}

  std::vector<boost::shared_ptr<QueryLabel>> &Labels() { return labels_; }

  boost::shared_ptr<QueryLabel> &LabelOfSample(int idx) {
    if (idx >= labels_.size()) {
      LOG(FATAL) << "Access sample index out of bound";
      std::abort();
    }
    return labels_[idx];
  }

  void SetVocabulary(boost::shared_ptr<Vocabulary> &vocab) { vocab_ = vocab; }

  boost::shared_ptr<Vocabulary> &GetVocabulary() { return vocab_; }

  void AppendQueryLabel(boost::shared_ptr<QueryLabel> &label) {
    labels_.push_back(label);
  }

  void SetQueryLabel(int idx, boost::shared_ptr<QueryLabel> &label) {
    while (idx >= labels_.size()) {
      labels_.resize(labels_.size() * 2);
      LOG(INFO) << "Expand label data size";
    }
    labels_[idx] = label;
  }

  size_t GetLabelSize() { return vocab_->VocabSize(); }

  size_t NumSamples() { return labels_.size(); }

private:
  std::vector<boost::shared_ptr<QueryLabel>> labels_;
  boost::shared_ptr<Vocabulary> vocab_;
};

class NNQueryFeaturizer {
public:
  NNQueryFeaturizer(boost::shared_ptr<Vocabulary> &word,
                    boost::shared_ptr<Vocabulary> &label)
      : wordvocab_(*word), labelvocab_(*label) {}
  ~NNQueryFeaturizer() {}

  bool FeaturizeLine(const std::string &query, const std::string &label,
                     boost::shared_ptr<QueryFeature> &feat,
                     boost::shared_ptr<QueryLabel> &l) {
    if (query.empty() || label.empty())
      return false;
    int vocabsize = wordvocab_.VocabSize();
    int labelsize = labelvocab_.VocabSize();

    std::vector<std::string> words;
    words.clear();
    words.push_back(Vocabulary::kBeginOfDoc);
    Util::Split(query, words, " ", true);
    words.push_back(Vocabulary::kEndOfDoc);

    if (words.size() < 3)
      return false;
    feat->Feature().resize(words.size(), vocabsize);
    feat->Feature().reserve(Eigen::VectorXi::Constant(words.size(), 1));
    for (int i = 0; i < words.size(); ++i) {
      int index = wordvocab_.GetIndex(words[i]);
      feat->Feature().insert(i, index) = 1;
    }

    words.clear();
    Util::Split(label, words, " ", true);
    if (words.size() != 1) {
      LOG(ERROR) << "Multiple labels appeared";
      return false;
    }
    l->Label() = labelvocab_.GetIndex(words[0]);
    return true;
  }

  // featurize passed in file
  bool Featurize(boost::shared_ptr<NNQueryFeature> &feats,
                 boost::shared_ptr<NNQueryLabel> &labels,
                 const std::string &filepath) {
    namespace signal2 = boost::signals2::detail;
    if (!feats.get() || !labels.get()) {
      LOG(INFO) << "Feature or label is empty" << std::endl;
      return false;
    }

    std::ifstream src(filepath);
    if (!src.is_open()) {
      LOG(ERROR) << "Failed to open file " << filepath << std::endl;
      return false;
    }

    signal2::auto_buffer<char, signal2::store_n_objects<10 * 1024>> buffer(
        10 * 1024, '\0');
    src.getline(buffer.data(), buffer.size());
    std::vector<std::string> segments;
    int linecount = 0;
    while (src.good()) {
      ++linecount;
      segments.clear();
      Util::Split((const unsigned char *)buffer.data(),
                  std::strlen(buffer.data()), segments,
                  (const unsigned char *)"\t", false);
      if (segments.size() < 2) {
        LOG(WARNING) << "Line " << linecount << " format error";
      } else {
        boost::shared_ptr<NNModel::QueryFeature> feat =
            boost::make_shared<NNModel::QueryFeature>();
        boost::shared_ptr<NNModel::QueryLabel> label =
            boost::make_shared<NNModel::QueryLabel>();
        if (FeaturizeLine(segments[0], segments[1], feat, label)) {
          feats->AppendQueryFeature(feat);
          labels->AppendQueryLabel(label);
        } else {
          feat.reset();
          label.reset();
        }
      }
      src.getline(buffer.data(), buffer.size());
    }

    if (!src.eof()) {
      LOG(ERROR) << "Unexpected error happens, donot reach EOF" << std::endl;
      return false;
    }

    return true;
  }

private:
  const Vocabulary &wordvocab_;
  const Vocabulary &labelvocab_;
};
} // namespace NNModel

#endif // !__NN_QUERY_H__
