#pragma once

#ifndef __VOCABULARY_H__
#define __VOCABULARY_H__
#include <boost/log/trivial.hpp>
#include <boost/signals2/detail/auto_buffer.hpp>
#include <fstream>
#include <unordered_map>
#include <string>
#include "../lccrf/cedar.h"

namespace NNModel {
	
	class Vocabulary {
	public:

		const static int UNKNOWN = 0;
		const static std::string UNK;
		const static std::string BeginOfDoc;
		const static std::string EndOfDoc;

		Vocabulary(){
			word2index_.clear();
			index2word_.clear();
			size_ = 0;
		}
		~Vocabulary();

		static boost::shared_ptr<Vocabulary> BuildVocabularyWithFilter(const std::string& filepath, size_t count);
		static boost::shared_ptr<Vocabulary> BuildVocabulary(const std::string& filepath);

		int GetIndex(const std::string& word){
			int id = word2index_.exactMatchSearch<int>(word.c_str());
			if (id != cedar::da<int>::CEDAR_NO_VALUE){
				return id;
			}
			else return Vocabulary::UNKNOWN;
		}

		bool AddWord(const std::string& word, int index){
			int id = word2index_.exactMatchSearch<int>(word.c_str());
			if (id != cedar::da<int>::CEDAR_NO_VALUE){
				index2word_[index] = word;
				word2index_.update(word.c_str(), word.size(), index);
			}
			else {
				BOOST_LOG_TRIVIAL(error) << "word " << word << " has been aded";
				return false;
			}
		}

		cedar::da<int>& Word2IndexMapping(){
			return word2index_;
		}

		size_t VocabSize() {
			return word2index_.num_keys();
		}

		const std::string& GetWord(int index){
			if (index2word_.count(index))
				return index2word_[index];
			else return Vocabulary::UNK;
		}

	private:
		cedar::da<int> word2index_;
		std::unordered_map<int, std::string> index2word_;
		size_t size_;
	};

	const std::string Vocabulary::UNK = "<UNK>";
	const std::string Vocabulary::BeginOfDoc = "<s>";
	const std::string Vocabulary::EndOfDoc = "</s>";

	boost::shared_ptr<Vocabulary> Vocabulary::BuildVocabulary(const std::string& filepath){
		namespace signal = boost::signals2::detail;
		signal::auto_buffer<char, signal::store_n_objects<1024>> buffer(1024, '\0');
		boost::shared_ptr<Vocabulary> vocab;
		std::ifstream src(filepath);
		if (!src.is_open()){
			BOOST_LOG_TRIVIAL(error) << "Failed to open file " << filepath << std::endl;
			return vocab;
		}

		vocab.reset(new Vocabulary());
		if (vocab.get() == nullptr){
			BOOST_LOG_TRIVIAL(info) << "Allocate vocabulary failed";
			return vocab;
		}

		vocab->AddWord(Vocabulary::UNK, 0);
		char* ptr = nullptr;
		src.getline(buffer.data(), buffer.size());
		while (src.good()){
			ptr = std::strtok(buffer.data(), " ");
			int key = vocab->VocabSize();
			std::string tlabel(buffer.data());
			vocab->AddWord(tlabel, key);
			src.getline(buffer.data(), buffer.size());
		}
	}

	boost::shared_ptr<Vocabulary> Vocabulary::BuildVocabularyWithFilter(const std::string& filepath, size_t cutoff){
		namespace signal = boost::signals2::detail;
		boost::shared_ptr<Vocabulary> vocab;
		signal::auto_buffer<char, signal::store_n_objects<16 * 1024>> buffer(16 * 1024, '\0');

		cedar::da<int> rawvocab;
		std::ifstream src(filepath);
		if (!src.is_open()){
			BOOST_LOG_TRIVIAL(fatal) << "Faled to open file " << filepath << std::endl;
			return vocab;
		}

		vocab.reset(new Vocabulary());
		if (!vocab.get()){
			BOOST_LOG_TRIVIAL(fatal) << "Failed to allocate memory";
			std::abort();
		}

		src.getline(buffer.data, buffer.size);
		while (src.good()){

			rawvocab.update(Vocabulary::BeginOfDoc.c_str(), 3, 1);
			char* ptr = std::strtok(buffer.data(), " ");
			while (!ptr){
				rawvocab.update(ptr, std::strlen(ptr), 1);
				ptr = std::strtok(NULL, " ");
			}
			rawvocab.update(Vocabulary::EndOfDoc.c_str(), 4, 1);

			src.getline(buffer.data(), buffer.size());
		}
		
		vocab->AddWord(Vocabulary::UNK, 0);
		vocab->AddWord(Vocabulary::BeginOfDoc, 1);
		vocab->AddWord(Vocabulary::EndOfDoc, 2);

		// iterator over vocabulary
		size_t from(0), p(0);
		int featcnt;
		for (featcnt = rawvocab.begin(from, p); featcnt != cedar::da<int>::CEDAR_NO_PATH; featcnt = rawvocab.next(from, p)) {
			if (featcnt >= cutoff) {
				rawvocab.suffix(buffer.data(), p, from);
				std::string tword(buffer.data());
				int key = vocab->VocabSize();
				if (!vocab->AddWord(tword, key)){
					BOOST_LOG_TRIVIAL(info) << "Add word failed" << std::endl;
				}
			}
		}
	}
}

#endif // __VOCABULARY_H__