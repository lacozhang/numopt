#include <boost/signals2/detail/auto_buffer.hpp>
#include "vocabulary.h"

const std::string Vocabulary::kUNK = "<UNK>";
const std::string Vocabulary::kBeginOfDoc = "<s>";
const std::string Vocabulary::kEndOfDoc = "</s>";

boost::shared_ptr<Vocabulary> Vocabulary::BuildVocabForQuery(const std::string & filepath, size_t cutoff, bool hasbos, bool haseos) {
	namespace signal = boost::signals2::detail;
	boost::shared_ptr<Vocabulary> vocab;
	signal::auto_buffer<char, signal::store_n_objects<4 * 1024>> buffer(4 * 1024, '\0');

	cedar::da<int> rawvocab;
	std::ifstream src(filepath);
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "Faled to open file " << filepath << std::endl;
		return vocab;
	}

	vocab.reset(new Vocabulary());
	if (!vocab.get()) {
		BOOST_LOG_TRIVIAL(fatal) << "Failed to allocate memory";
		return vocab;
	}

	src.getline(buffer.data(), buffer.size());
	std::vector<std::string> words;
	while (src.good()) {
		if (hasbos)
			rawvocab.update(Vocabulary::kBeginOfDoc.c_str(), Vocabulary::kBeginOfDoc.size(), 1);

		words.clear();
		Util::Split(buffer.data(), std::strlen(buffer.data()), words, " ", true);
		for (auto& word : words) {
			rawvocab.update(word.c_str(), word.size(), 1);
		}

		if (haseos)
			rawvocab.update(Vocabulary::kEndOfDoc.c_str(), Vocabulary::kEndOfDoc.size(), 1);

		src.getline(buffer.data(), buffer.size());
	}

	if (!src.eof()) {
		BOOST_LOG_TRIVIAL(warning) << "Unexpected EOF";
	}

	vocab->AddWord(Vocabulary::kUNK, Vocabulary::kUNKID);
	vocab->AddWord(Vocabulary::kBeginOfDoc, Vocabulary::kBOSID);
	vocab->AddWord(Vocabulary::kEndOfDoc, Vocabulary::kEOSID);

	// iterator over vocabulary
	size_t from(0), p(0);
	int featcnt;
	for (featcnt = rawvocab.begin(from, p); featcnt != cedar::da<int>::CEDAR_NO_PATH; featcnt = rawvocab.next(from, p)) {
		if (featcnt >= cutoff) {
			rawvocab.suffix(buffer.data(), p, from);
			std::string tword(buffer.data());
			int key = vocab->VocabSize();
			if (!vocab->AddWord(tword, key)) {
				BOOST_LOG_TRIVIAL(info) << "Add word failed" << std::endl;
			}
		}
	}

	return vocab;
}

boost::shared_ptr<Vocabulary> Vocabulary::BuildVocabForLabel(const std::string & filepath) {
	namespace signal = boost::signals2::detail;
	signal::auto_buffer<char, signal::store_n_objects<1024>> buffer(1024, '\0');
	boost::shared_ptr<Vocabulary> vocab;
	std::ifstream src(filepath);
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(error) << "Failed to open file " << filepath << std::endl;
		return vocab;
	}

	vocab.reset(new Vocabulary());
	if (vocab.get() == nullptr) {
		BOOST_LOG_TRIVIAL(info) << "Allocate vocabulary failed";
		return vocab;
	}

	std::vector<std::string> labels;
	cedar::da<int> rawlabels;
	src.getline(buffer.data(), buffer.size());
	while (src.good()) {
		rawlabels.clear();
		Util::Split(buffer.data(), std::strlen(buffer.data()), labels, " \t", true);
		if (labels.empty() || labels.size() > 1) {
			BOOST_LOG_TRIVIAL(warning) << "Multiple labels appeared " << buffer.data();
		}
		else {
			rawlabels.update(labels[0].c_str(), labels[0].size());
		}
		src.getline(buffer.data(), buffer.size());
	}

	if (!src.eof()) {
		BOOST_LOG_TRIVIAL(warning) << "Unexpected EOF";
	}

	vocab->AddWord(Vocabulary::kUNK, 0);
	// iterator over vocabulary
	size_t from(0), p(0);
	int featcnt;
	for (featcnt = rawlabels.begin(from, p); featcnt != cedar::da<int>::CEDAR_NO_PATH; featcnt = rawlabels.next(from, p)) {
		rawlabels.suffix(buffer.data(), p, from);
		std::string tword(buffer.data());
		int key = vocab->VocabSize();
		if (!vocab->AddWord(tword, key)) {
			BOOST_LOG_TRIVIAL(info) << "Label " << tword << " appear multiple times";
		}
	}

	return vocab;
}
