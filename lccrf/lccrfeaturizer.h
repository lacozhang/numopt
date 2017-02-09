#pragma once
#ifndef __LCCRFEATURIZER_H__
#define __LCCRFEATURIZER_H__
#include <vector>
#include <string>
#include <regex>
#include "crftemplate.h"

#ifdef USE_PREFIX_TREE
#include "cedar.h"
#else
#include "cedar.h"
#endif // USE_PREFIX_TREE

class LccrFeaturizer
{
public:
	LccrFeaturizer(std::string crftemplates) : crftemplates_(crftemplates){
		ngram2id_.clear();
	}
	~LccrFeaturizer();
	bool AccumulateFeatures(const std::string& featsrc);
	bool FeaturizeSentence(const std::vector<std::string>& lines, std::vector<std::vector<int>>& feats);
	bool FeaturizeFile(const std::string& featsrc, const std::string& featbin);
	bool Save(const std::string& featprefix);
	bool Load(const std::string& featprefix);
private:
	bool FromLineToRawFeatures(const std::vector<std::string>& lines, std::vector<std::vector<std::string>>& rawfeats);

	cedar::da<int> ngram2id_;
	CrfTemplate crftemplates_;
	static const std::regex kWSRegex;
};

#endif // !__LCCRFEATURIZER_H__
