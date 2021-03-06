#pragma once
#ifndef __LCCRFEATURIZER_H__
#define __LCCRFEATURIZER_H__
#include "crftemplate.h"
#include <regex>
#include <string>
#include <vector>

#ifdef USE_PREFIX_TREE
#include "util/cedar.h"
#else
#include "util/cedar.h"
#endif // USE_PREFIX_TREE

class LccrFeaturizer {
public:
  typedef cedar::da<int> trie_t;

  LccrFeaturizer(std::string crftemplates) : crftemplates_(crftemplates) {
    unifeat2id_.clear();
    bifeat2id_.clear();
    label2id_.clear();
  }

  ~LccrFeaturizer() {}

  bool AccumulateFeatures(const std::string &featsrc, int unifeatcut = 1,
                          int bifeatcut = 1);
  bool FeaturizeSentence(const std::vector<std::string> &lines,
                         std::vector<std::vector<int>> &unigramfeats,
                         std::vector<std::vector<int>> &bigramfeats,
                         std::vector<int> &labels);
  bool FeaturizeFile(const std::string &featsrc, const std::string &featbin);
  bool Save(const std::string &featprefix);
  bool Load(const std::string &featprefix);

private:
  bool FromLineToRawFeatures(const std::vector<std::string> &lines,
                             std::vector<std::vector<std::string>> &rawfeats,
                             std::vector<std::string> &rawlabels);
  bool AccumulateFeatureCountFromLine(
      trie_t &trie, const std::vector<std::vector<std::string>> &textfeats);
  bool AccumulateLabelFromLine(const std::vector<std::string> &textlabels);
  void FilterFeatureWithCount(trie_t &raw, trie_t &filtered, int cutoff);
  int CountSamples(const std::string &featsrc);
  bool ReadOneSentence(std::ifstream &src, std::vector<std::string> &sentence);
  bool FeaturizeWithTrie(const trie_t &trie,
                         const std::vector<std::string> &textfeats,
                         std::vector<int> &feats);

  cedar::da<int> unifeat2id_;
  cedar::da<int> bifeat2id_;
  cedar::da<int> label2id_;
  CrfTemplate crftemplates_;

  static const unsigned int MAXFEATLEN = 1024 * 10;
};

#endif // !__LCCRFEATURIZER_H__
