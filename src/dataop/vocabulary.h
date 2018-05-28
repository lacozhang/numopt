#pragma once

#ifndef __VOCABULARY_H__
#define __VOCABULARY_H__
#include <boost/make_shared.hpp>
#include <boost/log/trivial.hpp>
#include <fstream>
#include <unordered_map>
#include <string>
#include "../lccrf/cedar.h"
#include "../util/stringop.h"

class Vocabulary {
public:

	const static int kUNKID = 0;
	const static int kBOSID = 1;
	const static int kEOSID = 2;
	const static std::string kUNK;
	const static std::string kBeginOfDoc;
	const static std::string kEndOfDoc;

	Vocabulary(){
		word2index_.clear();
		index2word_.clear();
	}
	~Vocabulary(){}

	static boost::shared_ptr<Vocabulary> BuildVocabForQuery(const std::string& filepath, size_t cutoff, bool hasbos = true, bool haseos = true);

	static boost::shared_ptr<Vocabulary> BuildVocabForLabel(const std::string& filepath);

	int GetIndex(const std::string& word) const {
		int id = word2index_.exactMatchSearch<int>(word.c_str());
		if (id != cedar::da<int>::CEDAR_NO_VALUE){
			return id;
		}
		else return Vocabulary::kUNKID;
	}

	bool AddWord(const std::string& word, int index) {
		int id = word2index_.exactMatchSearch<int>(word.c_str());
		if (id == cedar::da<int>::CEDAR_NO_VALUE) {
			index2word_[index] = word;
			word2index_.update(word.c_str(), word.size()) = index;
			return true;
		}
		else {
			BOOST_LOG_TRIVIAL(error) << "word " << word << " has been aded";
			return false;
		}
	}

	size_t VocabSize() const {
		return word2index_.num_keys();
	}

	const std::string& GetWord(int index) {
		if (index2word_.count(index))
			return index2word_[index];
		else return Vocabulary::kUNK;
	}

private:
	cedar::da<int> word2index_;
	std::unordered_map<int, std::string> index2word_;
};

#endif // __VOCABULARY_H__