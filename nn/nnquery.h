#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/signals2/detail/auto_buffer.hpp>
#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>
#include "../typedef.h"
#include "../dataop/vocabulary.h"
#include "../util/stringop.h"

#ifndef __NN_QUERY_H__
#define __NN_QUERY_H__

namespace NNModel {
	class QueryFeature {
	public:
		QueryFeature(){}
		~QueryFeature(){}

		DataSamples& Feature(){
			return features_;
		}

	private:
		DataSamples features_;
	};

	class QueryLabel {
	public:
		QueryLabel(){}
		~QueryLabel(){}

		int& Label(){
			return labels_;
		}

	private:
		int labels_;
	};

	class NNQueryFeature {
	public:
		NNQueryFeature(){
			featdat_.clear();
		}
		~NNQueryFeature(){
		}

		std::vector<boost::shared_ptr<QueryFeature>>& Features(){
			return featdat_;
		}

		void SetVocabulary(boost::shared_ptr<Vocabulary>& vocab){
			vocab_ = vocab;
		}

		boost::shared_ptr<Vocabulary>& GetVocabulary(){
			return vocab_;
		}

		size_t GetVocabularySize(){
			return vocab_->VocabSize();
		}

		boost::shared_ptr<QueryFeature>& FeatureOfSample(int idx){
			if (idx >= featdat_.size()){
				BOOST_LOG_TRIVIAL(fatal) << "Access Sample Index out of bound";
			}
			return featdat_[idx];
		}

		size_t NumSamples(){
			return featdat_.size();
		}

	private:
		std::vector<boost::shared_ptr<QueryFeature>> featdat_;
		boost::shared_ptr<Vocabulary> vocab_;
	};

	class NNQueryLabel {
	public:
		NNQueryLabel(){
			labels_.clear();
		}
		~NNQueryLabel(){

		}

		std::vector<boost::shared_ptr<QueryLabel>>& Labels(){
			return labels_;
		}

		boost::shared_ptr<QueryLabel>& LabelOfSample(int idx){
			if (idx >= labels_.size()){
				BOOST_LOG_TRIVIAL(fatal) << "Access sample index out of bound";
			}
			return labels_[idx];
		}

		void SetVocabulary(boost::shared_ptr<Vocabulary>& vocab){
			vocab_ = vocab;
		}

		boost::shared_ptr<Vocabulary>& GetVocabulary(){
			return vocab_;
		}

		size_t GetLabelSize(){
			return vocab_->VocabSize();
		}

		size_t NumSamples(){
			return labels_.size();
		}

	private:
		std::vector<boost::shared_ptr<QueryLabel>> labels_;
		boost::shared_ptr<Vocabulary> vocab_;
	};

	class NNQueryFeaturizer{
	public:
		NNQueryFeaturizer(boost::shared_ptr<Vocabulary>& word, boost::shared_ptr<Vocabulary>& label)
			:wordvocab_(*word), labelvocab_(*label){
		}
		~NNQueryFeaturizer(){}

		bool FeaturizeLine(const std::string& query, const std::string& label,
			boost::shared_ptr<QueryFeature>& feat, boost::shared_ptr<QueryLabel>& l){
			int vocabsize = wordvocab_.VocabSize();
			int labelsize = wordvocab_.VocabSize();

			std::vector<std::string> words;
			words.push_back(Vocabulary::BeginOfDoc);
			Split(query, words, " ", true);
			words.push_back(Vocabulary::EndOfDoc);

			if (words.size() < 3) return false;
			for (int i = 0; i < words.size(); ++i){
				int index = wordvocab_.GetIndex(words[i]);
				feat->Feature().insert(i, index) = 1;
			}

			words.clear();
			Split(label, words, " ", true);
			if (words.size() != 1){
				BOOST_LOG_TRIVIAL(error) << "Multiple labels appeared";
				return false;
			}
			l->Label() = labelvocab_.GetIndex(words[0]);
			return true;
		}
		
		// featurize passed in file
		bool Featurize(boost::shared_ptr<NNQueryFeature>& feats, boost::shared_ptr<NNQueryLabel>& labels,
			const std::string& filepath){
			namespace signal2 = boost::signals2::detail;
			if (!feats.get() || !labels.get()){
				BOOST_LOG_TRIVIAL(info) << "Feature or label is empty" << std::endl;
				return false;
			}

			std::ifstream src(filepath);
			if (!src.is_open()){
				BOOST_LOG_TRIVIAL(error) << "Failed to open file " << filepath << std::endl;
				return false;
			}

			signal2::auto_buffer<char, signal2::store_n_objects<10 * 1024>> buffer(10 * 1024, '\0');
			src.getline(buffer.data(), buffer.size());
			char* ptr = nullptr;
			while (src.good()){
				ptr = std::strtok(buffer.data(), "\t");
				if (ptr != nullptr){
					std::string query(ptr);

					ptr = std::strtok(NULL, "\t");
					if (ptr == nullptr){
						BOOST_LOG_TRIVIAL(error) << "Line no Label";
					}
					else {
						std::string label(ptr);
						boost::shared_ptr<QueryFeature> feat;
						boost::shared_ptr<QueryLabel> featlabel;
						if (!FeaturizeLine(query, label, feat, featlabel)){
							feats->Features().push_back(feat);
							labels->Labels().push_back(featlabel);
						}
					}
				}
				else {
					BOOST_LOG_TRIVIAL(info) << "Empty line";
				}

				src.getline(buffer.data(), buffer.size());
			}

			if (!src.eof()){
				BOOST_LOG_TRIVIAL(error) << "Unexpected error happens, donot reach EOF" << std::endl;
			}
		}

	private:
		Vocabulary& wordvocab_;
		Vocabulary& labelvocab_;
	};
}

#endif // !__NN_QUERY_H__
