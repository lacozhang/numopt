#include <iostream>
#include <fstream>
#include <string>
#include <cctype>
#include <boost/signals2/detail/auto_buffer.hpp>
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "lccrfeaturizer.h"
#include "../util.h"

bool LccrFeaturizer::AccumulateFeatures(const std::string& featsrc, int unifeatcut, int bifeatcut) {
	namespace fs = boost::filesystem;

	BOOST_ASSERT_MSG(unifeatcut >= 1, "unigram feature cutoff value less than 1");
	BOOST_ASSERT_MSG(bifeatcut >= 1, "bigram feature cutoff value less than 1");

	if (!crftemplates_.IsValid()) {
		BOOST_LOG_TRIVIAL(fatal) << "Crf templates load failed";
		return false;
	}

	fs::path filepath(featsrc);
	if (!fs::exists(filepath) || !fs::is_regular_file(filepath)) {
		BOOST_LOG_TRIVIAL(fatal) << "Source file " << featsrc
			<< " do no exists or is not a regular file";
		return false;
	}

	std::ifstream src(featsrc);
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "Open file " << featsrc
			<< " failed";
		return false;
	}
	std::vector<std::string> sentence;
	std::vector<std::string> rawlabels;
	std::vector<std::vector<std::string>> rawtexts;
	std::vector<std::vector<std::string>> textfeats;
	trie_t unifeatcache, bifeatcache;
	unifeatcache.clear();
	bifeatcache.clear();

	sentence.clear();
	ReadOneSentence(src, sentence);
	while (src.good() || (sentence.size() > 0)) {

		if (sentence.size() > 0) {
			FromLineToRawFeatures(sentence, rawtexts, rawlabels);
			crftemplates_.ExtractUnigramFeatures(rawtexts, textfeats);
			AccumulateFeatureCountFromLine(unifeatcache, textfeats);
			crftemplates_.ExtractBigramFeatures(rawtexts, textfeats);
			AccumulateFeatureCountFromLine(bifeatcache, textfeats);
			AccumulateLabelFromLine(rawlabels);
			sentence.clear();
		}

		ReadOneSentence(src, sentence);
	}

	if (!src.eof()) {
		BOOST_LOG_TRIVIAL(error) << "file " << featsrc << " ended unexpectedly";
		return false;
	}
	src.close();

	FilterFeatureWithCount(unifeatcache, unifeat2id_, unifeatcut);
	FilterFeatureWithCount(bifeatcache, bifeat2id_, bifeatcut);
	return true;
}

bool LccrFeaturizer::FeaturizeSentence(
	const std::vector<std::string>& lines,
	std::vector<std::vector<int>>& unigramfeats,
	std::vector<std::vector<int>>& bigramfeats,
	std::vector<int>& labels)
{
	unigramfeats.clear();
	bigramfeats.clear();
	labels.clear();

	if (lines.size() < 1) {
		return false;
	}

	std::vector<std::vector<std::string>> rawtexts;
	std::vector<std::vector<std::string>> textfeats;
	std::vector<std::string> rawlabels;
	std::vector<int> featids;
	int labelid;

	FromLineToRawFeatures(lines, rawtexts, rawlabels);
	crftemplates_.ExtractUnigramFeatures(rawtexts, textfeats);
	for (const std::vector<std::string>& feat : textfeats) {
		FeaturizeWithTrie(unifeat2id_, feat, featids);

#ifdef _DEBUG
		std::sort(featids.begin(), featids.end());
		bool duplicatefeature = false;
		for (int i = 0; i < featids.size() - 1; ++i) {
			if (featids[i] == featids[i + 1]) {
				duplicatefeature = true;
			}
		}
		BOOST_ASSERT(!duplicatefeature);

#endif // _DEBUG

		unigramfeats.push_back(featids);
	}


	crftemplates_.ExtractBigramFeatures(rawtexts, textfeats);
	for (const std::vector<std::string>& feat : textfeats) {
		FeaturizeWithTrie(bifeat2id_, feat, featids);
#ifdef _DEBUG
		std::sort(featids.begin(), featids.end());
		bool duplicatefeature = false;
		for (int i = 0; i < featids.size() - 1; ++i) {
			if (featids[i] == featids[i + 1]) {
				duplicatefeature = true;
			}
		}
		BOOST_ASSERT(!duplicatefeature);
#endif // _DEBUG
		bigramfeats.push_back(featids);
	}



	for (const std::string& label : rawlabels) {
		labelid = label2id_.exactMatchSearch<int>(label.c_str());
		if (cedar::da<int>::CEDAR_NO_VALUE != labelid) {
			labels.push_back(labelid);
		}
		else {
			BOOST_LOG_TRIVIAL(fatal) << "unrecognized label " << label;
		}
	}

	return true;
}

bool LccrFeaturizer::FeaturizeFile(const std::string & featsrc, const std::string & featbin)
{
	namespace fs = boost::filesystem;
	fs::path textfilepath(featsrc);
	if (!fs::exists(textfilepath) || !fs::is_regular_file(textfilepath)) {
		BOOST_LOG_TRIVIAL(fatal) << " file " << featsrc << " not exist or is not a file";
		return false;
	}

	std::ifstream src(featsrc);
	std::fstream sink(featbin, std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);

	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "open " << featsrc << " failed";
		return false;
	}
	if (!sink.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "open " << featbin << " failed";
		return false;
	}

	int numsentences = CountSamples(featsrc);
	int maxunifeatid = unifeat2id_.num_keys() - 1;
	int maxbifeatid = bifeat2id_.num_keys() - 1;
	int maxlabelid = label2id_.num_keys() - 1;

	BOOST_LOG_TRIVIAL(info) << "Samples             : " << numsentences;
	BOOST_LOG_TRIVIAL(info) << "Max Unigram Feat Id : " << maxunifeatid;
	BOOST_LOG_TRIVIAL(info) << "Max Bigram Feat Id  : " << maxbifeatid;
	BOOST_LOG_TRIVIAL(info) << "Max Label Id        : " << maxlabelid;

	std::vector<std::string> sentence;

	std::vector<std::vector<int>> unigramfeats;
	std::vector<std::vector<int>> bigramfeats;
	std::vector<int> labels;

	BinaryFileHandler writer(sink);
	writer.WriteInt(maxunifeatid);
	writer.WriteInt(maxbifeatid);
	writer.WriteInt(maxlabelid);
	writer.WriteInt(numsentences);

	ReadOneSentence(src, sentence);
	while (src.good() || (sentence.size() > 0))
	{
		if (sentence.size() > 0) {
			FeaturizeSentence(sentence, unigramfeats, bigramfeats, labels);
			BOOST_ASSERT_MSG(sentence.size() == unigramfeats.size(), "number not equal");
			BOOST_ASSERT_MSG(sentence.size() == bigramfeats.size(), "number not equal");
			BOOST_ASSERT_MSG(sentence.size() == labels.size(), "number not equal");
			size_t wordcount = sentence.size();
			writer.WriteSizeT(wordcount);

			for (size_t i = 0; i < wordcount; ++i) {
				int labelid = labels[i];
				int unigramfeatcnt = unigramfeats[i].size();
				int bigramfeatcnt = bigramfeats[i].size();
				writer.WriteInt(labelid);
				writer.WriteInt(unigramfeatcnt);
				for (const int& featid : unigramfeats[i]) {
					writer.WriteInt(featid);
				}
				writer.WriteInt(bigramfeatcnt);
				for (const int& featid : bigramfeats[i]) {
					writer.WriteInt(featid);
				}
			}
			sentence.clear();
		}

		ReadOneSentence(src, sentence);
	}

	src.close();
	sink.close();
	return true;
}

bool LccrFeaturizer::Save(const std::string & featprefix)
{
	std::string unifeatpath = featprefix + ".1gram.bin";
	std::string bifeatpath = featprefix + ".2gram.bin";
	std::string label2idpath = featprefix + ".label.bin";
	std::string templatepath = featprefix + ".template.txt";

	unifeat2id_.save(unifeatpath.c_str());
	bifeat2id_.save(bifeatpath.c_str());
	label2id_.save(label2idpath.c_str());
	crftemplates_.SaveToFile(templatepath);
	return true;
}

bool LccrFeaturizer::Load(const std::string & featprefix)
{
	std::string unifeatpath = featprefix + ".1gram.bin";
	std::string bifeatpath = featprefix + ".2gram.bin";
	std::string label2idpath = featprefix + ".label.bin";
	std::string templatepath = featprefix + ".template.txt";

	unifeat2id_.open(unifeatpath.c_str());
	bifeat2id_.open(bifeatpath.c_str());
	label2id_.open(label2idpath.c_str());
	crftemplates_.LoadTemplate(templatepath);

	BOOST_LOG_TRIVIAL(info) << "Max Unigram Feat Id : " << unifeat2id_.num_keys() - 1;
	BOOST_LOG_TRIVIAL(info) << "Max Bigram Feat Id  : " << bifeat2id_.num_keys() - 1;
	BOOST_LOG_TRIVIAL(info) << "Max Label Id        : " << label2id_.num_keys() - 1;

	return true;
}


bool LccrFeaturizer::FromLineToRawFeatures(
	const std::vector<std::string>& lines,
	std::vector<std::vector<std::string>>& rawfeats, std::vector<std::string>& rawlabels) {

	rawfeats.clear();
	rawlabels.clear();

	for (const std::string& line : lines) {
		std::string temp;
		std::vector<std::string> segs;
		temp.clear();
		segs.clear();

		for (int i = 0; i < line.size(); ++i) {
			if (std::isspace(line[i])) {
				if (temp.size() > 0) {
					segs.push_back(temp);
					temp.clear();
				}
			}
			else {
				temp.push_back(line[i]);
			}
		}

		if (temp.size() < 1) {
			BOOST_LOG_TRIVIAL(error) << "line format error " << line;
		}
		else {
			rawlabels.push_back(temp);
		}

		if (segs.size() < 1) {
			BOOST_LOG_TRIVIAL(warning) << "extract nothing, maybe bad line " << line;
		}
		rawfeats.push_back(segs);
	}
	return true;
}

bool LccrFeaturizer::AccumulateFeatureCountFromLine(trie_t& trie, const std::vector<std::vector<std::string>>& textfeats)
{
	for (const std::vector<std::string>& posfeat : textfeats) {
		for (const std::string& feat : posfeat) {
			trie.update(feat.c_str(), feat.size(), 1);
		}
	}
	return true;
}

bool LccrFeaturizer::AccumulateLabelFromLine(const std::vector<std::string>& textlabels)
{
	for (const std::string& label : textlabels) {
		if (cedar::da<int>::CEDAR_NO_VALUE == label2id_.exactMatchSearch<int>(label.c_str(), label.size())) {
			int keys = label2id_.num_keys();
			label2id_.update(label.c_str(), label.size(), keys);
		}
	}
	return true;
}

void LccrFeaturizer::FilterFeatureWithCount(trie_t & raw, trie_t & filtered, int cutoff)
{
	namespace signal = boost::signals2::detail;
	size_t from(0), p(0);
	signal::auto_buffer<char, signal::store_n_bytes<LccrFeaturizer::MAXFEATLEN>> buffer(LccrFeaturizer::MAXFEATLEN, '\0');
	int featcnt;
	for (featcnt = raw.begin(from, p); featcnt != trie_t::CEDAR_NO_PATH; featcnt = raw.next(from, p)) {
		if (featcnt >= cutoff) {
			raw.suffix(buffer.data(), p, from);
			int key = filtered.num_keys();
			filtered.update(buffer.data(), p, 0) = key;
		}
	}
}


int LccrFeaturizer::CountSamples(const std::string & featsrc)
{
	std::ifstream src(featsrc);
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "failed to open file " << featsrc;
		return -1;
	}

	std::vector<std::string> sentence;
	int count = 0;
	ReadOneSentence(src, sentence);
	while (src.good() || (sentence.size() > 0))
	{
		if (sentence.size() > 0) {
			count++;
			sentence.clear();
		}
		ReadOneSentence(src, sentence);
	}

	if (!src.eof()) {
		BOOST_LOG_TRIVIAL(fatal) << "unexpected EOF " << __FILE__ << " " << __LINE__;
	}
	return count;
}

bool LccrFeaturizer::ReadOneSentence(std::ifstream & src, std::vector<std::string>& sentence)
{
	if (!src.good()) {
		BOOST_LOG_TRIVIAL(info) << "stream in bad state";
		return false;
	}

	sentence.clear();
	std::string line;

	std::getline(src, line);
	boost::algorithm::trim(line);
	while (src.good() && (line.size() < 1))
	{
		std::getline(src, line);
		boost::algorithm::trim(line);
	}
	
	if ((!src.good()) && (!src.eof())) {
		BOOST_LOG_TRIVIAL(fatal) << "error when read files";
		return false;
	}
	
	while (src.good() && line.size() > 0)
	{
		sentence.push_back(line);
		std::getline(src, line);
		boost::algorithm::trim(line);
	}

	return true;
}

bool LccrFeaturizer::FeaturizeWithTrie(const trie_t& trie, const std::vector<std::string>& textfeats, std::vector<int>& feats)
{
	feats.clear();
	for (const std::string& feat : textfeats) {
		int featid = trie.exactMatchSearch<int>(feat.c_str());
		if (featid != trie_t::CEDAR_NO_VALUE) {
			feats.push_back(featid);
		}
	}
	return true;
}
