#pragma once
#ifndef __LCCRFEATURIZER_H__
#define __LCCRFEATURIZER_H__
#include <vector>
#include <string>

#ifdef USE_PREFIX_TREE
#include "cedar.h"
#else
#include "cedar.h"
#endif // USE_PREFIX_TREE

class LccrFeaturizer
{
public:
	LccrFeaturizer(std::string crftemplates);
	~LccrFeaturizer();
	bool FeaturizeFiles(std::vector<std::string>& featsrcs);
	bool Save(std::string featbin);
private:
	
	
	std::vector<std::string> featsrcs_;
	cedar::da<int> ngram2id_;

};

#endif // !__LCCRFEATURIZER_H__
