#pragma warning(disable : 4996)

#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <fstream>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>
#include "dataop.h"
#include "util.h"

namespace {

	static const char* kBinMagicString = "binv2";
	static const char *libsvmseps = "\t ";
	static char TempLineBuffer[UINT16_MAX] = { '\0' };
}

void matrix_size_estimation_from_text(std::ifstream& featsrc, std::vector<size_t>& rowsize,
	int &row, int &col) {

	row = col = 0;

	std::vector<std::pair<size_t, float>> feats;
	int label = 0;

	featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));
	while (featsrc.good()) {
		++row;
		parselibsvmline(TempLineBuffer, feats, label, true);
		// active feature for sample row
		rowsize.push_back(feats.size() + 1);

		for (std::pair<size_t, float> &item : feats) {
			if (col < item.first) {
				col = item.first;
			}
		}

		// get next line from file
		featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));
	}
}


bool matrix_size_estimation_from_bin(std::ifstream& ifs, std::vector<size_t>& rowsize,
	int &row, int &col) {
	row = col = 0;

	bool success = true;
	size_t featindex = 0, samplecnt = 0, featcnt = 0, elementcnt = 0;
	float featval = 0;
	size_t temp = 0;
	signed char label = 0;

	// read magic string
	std::memset(TempLineBuffer, 0, sizeof(TempLineBuffer));
	ifs.read(TempLineBuffer, std::strlen(kBinMagicString));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read magic string failed";
		return false;
	}
	if (std::strncmp(TempLineBuffer, kBinMagicString, std::strlen(kBinMagicString)) != 0) {
		BOOST_LOG_TRIVIAL(error) << "magic string do not equal";
		return false;
	}

	// read sample count
	ifs.read((char*)&samplecnt, sizeof(size_t));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read sample count failed";
		return false;
	}
	row = samplecnt;
	rowsize.resize(row);

	// read maximum feature id
	ifs.read((char*)&featcnt, sizeof(size_t));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read maximum feature id failed";
		return false;
	}
	col = featcnt;

	// read non-zero elements in data file
	ifs.read((char*)&elementcnt, sizeof(size_t));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read number of non-zero elements failed";
		return false;
	}

	// read number of non-zero elements in each row.
	for (int rowindex = 0; rowindex < row; ++rowindex) {
		ifs.read((char*)&temp, sizeof(size_t));
		success = success && ifs.good();
		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "read non-zero elements for each row failed";
			return false;
		}
		rowsize[rowindex] = temp;
	}
	return true;
}

bool load_libsvm_data_text(std::ifstream& ifs, boost::shared_ptr<DataSamples> samples,
	boost::shared_ptr<LabelVector> labels, int featsize) {

	std::vector<std::pair<size_t, float>> featline;

	ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
	int nrow = 0;
	while (ifs.good()) {

		int label;
		parselibsvmline(TempLineBuffer, featline, label, true);
		labels->coeffRef(nrow) = label;

		for (std::pair<size_t, float> &item : featline) {
			if (item.first < featsize) {
				samples->insert(nrow, item.first) = item.second;
			}
		}
		++nrow;
		ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
	}
	return true;
}

bool load_libsvm_data_bin(std::ifstream& ifs, boost::shared_ptr<DataSamples>& samples,
	boost::shared_ptr<LabelVector>& labels, int featsize) {

	bool success = true;
	int rowindex = 0;
	size_t featindex = 0, samplecnt = 0, featcnt = 0, elementcnt = 0;
	float featval = 0;
	size_t temp = 0;
	signed char label = 0;

	// read magic string
	std::memset(TempLineBuffer, 0, sizeof(TempLineBuffer));
	ifs.read(TempLineBuffer, std::strlen(kBinMagicString));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read magic string failed";
		return false;
	}
	if (std::strncmp(TempLineBuffer, kBinMagicString, std::strlen(kBinMagicString)) != 0) {
		BOOST_LOG_TRIVIAL(error) << "magic string do not equal";
		return false;
	}

	// read sample count
	ifs.read((char*)&samplecnt, sizeof(size_t));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read sample count failed";
		return false;
	}

	// read maximum feature id
	ifs.read((char*)&featcnt, sizeof(size_t));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read maximum feature id failed";
		return false;
	}

	if (featcnt > samples->cols()) {
		BOOST_LOG_TRIVIAL(info) << "maximum feature id of file: " << featcnt
			<< "maximum feature id of data matrix: " << samples->cols();
	}

	// read non-zero elements in data file
	ifs.read((char*)&elementcnt, sizeof(size_t));
	success = success && ifs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "read number of non-zero elements failed";
		return false;
	}

	// read number of non-zero elements in each row.
	for (int rowindex = 0; rowindex < samplecnt; ++rowindex) {
		ifs.read((char*)&temp, sizeof(size_t));
		success = success && ifs.good();
		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "read non-zero elements for each row failed";
			return false;
		}
	}

	// read real data
	for (rowindex = 0; rowindex < samplecnt; ++rowindex) {
		ifs.read((char*)&temp, sizeof(size_t));
		success = success && ifs.good();
		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "read non-zero elements of line failed";
			return false;
		}

		ifs.read((char*)&label, sizeof(signed char));
		success = success && ifs.good();
		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "read label of line failed";
			return false;
		}

		labels->coeffRef(rowindex) = label;

		for (int i = 0; i < temp; ++i) {

			ifs.read((char*)&featindex, sizeof(size_t));
			success = success && ifs.good();
			if (!success) {
				BOOST_LOG_TRIVIAL(error) << "read index of feat failed";
				return false;
			}

			ifs.read((char*)&featval, sizeof(float));
			success = success && ifs.good();
			if (!success) {
				BOOST_LOG_TRIVIAL(error) << "read value of feat failed";
				return false;
			}

			if (featindex <= samples->cols()) {
				samples->coeffRef(rowindex, featindex) = featval;
			}
		}
	}
	std::cout << "Load Total " << rowindex << " samples" << std::endl;
	return true;
}

bool save_libsvm_data_bin(std::string filepath,
	boost::shared_ptr<DataSamples>& samples,
	boost::shared_ptr<LabelVector>& labels) {

	if (!boost::algorithm::ends_with(filepath, ".bin")) {
		BOOST_LOG_TRIVIAL(trace) << "output file do not ends with .bin";
	}

	std::ofstream ofs(filepath, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
	if (!ofs.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "create file " << filepath << " failed";
		return false;
	}

	size_t samplecnt = samples->rows(), featurecnt = samples->cols() - 1, elementcnt = samples->nonZeros();
	bool success = true;

	// write magic string
	ofs.write(kBinMagicString, std::strlen(kBinMagicString));
	success = success && ofs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "write magic number failed";
		return false;
	}

	// write number of samples in this set
	ofs.write((char*)&samplecnt, sizeof(size_t));
	success = success && ofs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "write sample count failed";
		return false;
	}

	// write maximum feature id in this set
	ofs.write((const char*)&featurecnt, sizeof(size_t));
	success = success && ofs.good();
	if (!success) {
		BOOST_LOG_TRIVIAL(error) << "write maximum feature id failed";
		return false;
	}

	// write number of non-zero elements in this set
	ofs.write((const char*)&elementcnt, sizeof(size_t));
	success = success && ofs.good();

	// write number of non-zero elements in each row
	for (int rowindex = 0; rowindex < samples->outerSize(); ++rowindex) {
		size_t cnt = 0;
		for (DataSamples::InnerIterator it(*samples, rowindex); it; ++it, ++cnt);

		ofs.write((char*)&cnt, sizeof(size_t));
		success = success && ofs.good();
		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "write nonzero elements of one row failed";
			return false;
		}
	}

	// write data labels into file
	BOOST_LOG_TRIVIAL(trace) << "write data to file";
	signed char label = 0;
	size_t index = 0;
	float val = 0;

	for (int rowindex = 0; rowindex < samples->outerSize(); ++rowindex) {

		size_t cnt = 0;
		for (DataSamples::InnerIterator it(*samples, rowindex); it; ++it, ++cnt);

		ofs.write((char*)&cnt, sizeof(size_t));
		success = success && ofs.good();
		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "write element count failed";
			return false;
		}

		label = labels->coeff(rowindex);
		ofs.write((char*)&label, sizeof(signed char));
		success = success && ofs.good();
		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "write line " << rowindex << " label failed";
			return false;
		}

		if (!success) {
			BOOST_LOG_TRIVIAL(error) << "write label failed";
			return false;
		}

		for (DataSamples::InnerIterator it(*samples, rowindex); it; ++it) {
			index = it.col();
			val = it.value();

			ofs.write((char*)&index, sizeof(size_t));
			success = success && ofs.good();

			if (!success) {
				BOOST_LOG_TRIVIAL(fatal) << "write index for line " << rowindex << " failed";
				return false;
			}

			ofs.write((char*)&val, sizeof(float));
			success = success && ofs.good();
			if (!success) {
				BOOST_LOG_TRIVIAL(fatal) << "write value for index line " << rowindex << " failed";
				return false;
			}
		}
	}
	ofs.close();
	BOOST_LOG_TRIVIAL(info) << "sample count  : " << samplecnt;
	BOOST_LOG_TRIVIAL(info) << "max feature id: " << featurecnt;
	return true;
}

void parselibsvmline(char *line, std::vector<std::pair<size_t, float>> &feats,
	int &label, bool parse) {
	feats.clear();

	char *ptr = strtok(line, libsvmseps);
	if (!ptr) {
		std::cerr << "Error, string abnormal" << line << std::endl;
		std::exit(-1);
	}
	label = std::atoi(ptr);

	ptr = strtok(NULL, ": \t");
	while (ptr != nullptr) {
		size_t index = std::atoi(ptr);
		float val = 0;

		ptr = strtok(NULL, ": \t");
		if (ptr != nullptr) {
			val = std::atof(ptr);
			ptr = strtok(NULL, ": \t");
			feats.push_back(std::pair<size_t, float>(index, val));
		}
		else {
			BOOST_LOG_TRIVIAL(error) << "error, data format error" << std::endl;
			std::abort();
		}
	}
}

bool matrix_size_estimation(std::string featfile, Eigen::VectorXi &datsize,
	int &row, int &col) {

	bool filebinary = false;
	if (boost::algorithm::ends_with(featfile, ".bin")) {
		filebinary = true;
	}

	std::ifstream featsrc;
	if (!filebinary) {
		featsrc.open(featfile.c_str(), std::ios_base::in);
	}
	else {
		featsrc.open(featfile.c_str(), std::ios_base::in | std::ios_base::binary);
	}

	if (!featsrc.is_open()) {
		BOOST_LOG_TRIVIAL(info) << "open file " << featfile << " failed" << std::endl;
		std::abort();
	}

	timeutil t;
	t.tic();

	std::vector<size_t> rowsize;
	if (!filebinary) {
		matrix_size_estimation_from_text(featsrc, rowsize, row, col);
	}
	else {
		if (!matrix_size_estimation_from_bin(featsrc, rowsize, row, col)) {
			BOOST_LOG_TRIVIAL(fatal) << "estimate matrix size failed " << featfile;
			return false;
		}
	}

	featsrc.close();
	BOOST_LOG_TRIVIAL(info) << "estimate costs " << t.toc() << " seconds";

	col += 1;

	datsize.resize(row);
	for (int i = 0; i < rowsize.size(); ++i) {
		datsize(i) = rowsize[i];
	}
	return true;
}

bool load_libsvm_data(
	std::string featfile,
	boost::shared_ptr<DataSamples> &Samples,
	boost::shared_ptr<LabelVector> &labels, bool estimate, int colsize) {

	// estimate the data size for loading
	Eigen::VectorXi datasize;
	std::vector<std::pair<size_t, double>> featline;

	bool binary = boost::algorithm::ends_with(featfile, ".bin");
	if (binary) {
		BOOST_LOG_TRIVIAL(info) << "Loading data from binary file";
	}
	else {
		BOOST_LOG_TRIVIAL(info) << "Loading data from text file";
	}

	int estrowsize, estcolsize;
	if (!matrix_size_estimation(featfile, datasize, estrowsize, estcolsize)) {
		BOOST_LOG_TRIVIAL(fatal) << "load data file " << featfile << " failed";
		return false;
	}

	BOOST_LOG_TRIVIAL(trace) << "Finish data size estimation";

	timeutil t;

	if (estimate)
		colsize = estcolsize;

	BOOST_LOG_TRIVIAL(trace) << "Sample count  :" << estrowsize;
	BOOST_LOG_TRIVIAL(trace) << "Max feature id:" << (colsize - 1);
	Samples.reset(new DataSamples(estrowsize, colsize));

	if (Samples.get() == NULL) {
		BOOST_LOG_TRIVIAL(error) << "Error, new operator for samples error" << std::endl;
		std::exit(-1);
	}

	Samples->reserve(datasize);

	labels.reset(new LabelVector(estrowsize));

	std::ifstream featsrc;
	if (binary) {
		featsrc.open(featfile, std::ios_base::in | std::ios_base::binary);
	}
	else {
		featsrc.open(featfile, std::ios_base::in);
	}

	if (!featsrc.is_open()) {
		BOOST_LOG_TRIVIAL(error) << "open file " << featfile << " failed";
		return false;
	}

	bool ret = true;
	t.tic();
	if (binary) {
		ret = load_libsvm_data_bin(featsrc, Samples, labels, colsize);
	}
	else {
		ret = load_libsvm_data_text(featsrc, Samples, labels, colsize);
	}

	Samples->makeCompressed();
	BOOST_LOG_TRIVIAL(info) << "Loading data costs " << t.toc() << " seconds " << std::endl;
	BOOST_LOG_TRIVIAL(info) << "data samples : " << Samples->rows() << std::endl;
	BOOST_LOG_TRIVIAL(info) << "max feat id  : " << (Samples->cols() - 1) << std::endl;
	return ret;
}

bool load_lccrf_data_bin(std::string srcfilepath,
	boost::shared_ptr<LccrfSamples>& samples,
	boost::shared_ptr<LccrfLabels>& labels,
	bool estimate,
	size_t maxunifeatid, size_t maxbifeatid, size_t maxlabelid) {

	bool binary = false;
	binary = boost::algorithm::ends_with(srcfilepath, ".bin");

	if (!binary) {
		BOOST_LOG_TRIVIAL(error) << "For lccrf, binary file only";
		return false;
	}

	std::vector<std::vector<int>> currunifeats, currbifeats;
	std::vector<int> currlabels, currunifeatcnts, currbifeatcnts;

	std::fstream src(srcfilepath, std::ios_base::in | std::ios_base::binary);
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(error) << "Failed to open file " << srcfilepath;
		return false;
	}

	BinaryFileHandler reader(src);
	int numsamples = 0;
	size_t wordcount;
	int tmp;

	if (!reader.ReadInt(tmp)) {
		BOOST_LOG_TRIVIAL(error) << "Error when read max unigram feature id";
		return false;
	}
	if (estimate) {
		maxunifeatid = tmp;
	}

	if (!reader.ReadInt(tmp)) {
		BOOST_LOG_TRIVIAL(error) << "Error when read max bigram feature id";
		return false;
	}
	if (estimate) {
		maxbifeatid = tmp;
	}

	if (!reader.ReadInt(tmp)) {
		BOOST_LOG_TRIVIAL(error) << "Error when read max label id";
		return false;
	}
	if (estimate) {
		maxlabelid = tmp;
	}
	
	if (!reader.ReadInt(numsamples)) {
		BOOST_LOG_TRIVIAL(error) << "Error when read number of samples";
		return false;
	}

	BOOST_LOG_TRIVIAL(info) << "Number of Samples      : " << numsamples;
	BOOST_LOG_TRIVIAL(info) << "Max Unigram Feature Id : " << maxunifeatid;
	BOOST_LOG_TRIVIAL(info) << "Max Bigram Feature Id  : " << maxbifeatid;
	BOOST_LOG_TRIVIAL(info) << "Max Label Id           : " << maxlabelid;

	samples.reset(new LccrfSamples());
	labels.reset(new LccrfLabels());

	samples->SetMaxUnigramFeatureId(maxunifeatid);
	samples->SetMaxBigramFeatureId(maxbifeatid);
	labels->SetMaxLabelId(maxlabelid);

	samples->UnigramFeature().resize(numsamples);
	samples->BigramFeature().resize(numsamples);
	labels->Labels().resize(numsamples);
	
	for (int sampleidx = 0; sampleidx < numsamples; ++sampleidx) {

		reader.ReadSizeT(wordcount);
		currunifeats.resize(wordcount);
		currbifeats.resize(wordcount);
		currlabels.resize(wordcount);
		currunifeatcnts.resize(wordcount);
		currbifeatcnts.resize(wordcount);

		for (size_t pos = 0; pos < wordcount; ++pos) {
			reader.ReadInt(currlabels[pos]);
			reader.ReadInt(currunifeatcnts[pos]);
			int featid;
			for (int idx = 0; idx < currunifeatcnts[pos]; ++idx) {
				reader.ReadInt(featid);
				currunifeats[pos].push_back(featid);
			}

			reader.ReadInt(currbifeatcnts[pos]);
			for (int idx = 0; idx < currbifeatcnts[pos]; ++idx) {
				reader.ReadInt(featid);
				currbifeats[pos].push_back(featid);
			}

			std::sort(currunifeats[pos].begin(), currunifeats[pos].end());
			std::sort(currbifeats[pos].begin(), currbifeats[pos].end());
		}

		samples->UnigramFeature()[sampleidx].reset(new DataSamples(wordcount, maxunifeatid + 1));
		samples->BigramFeature()[sampleidx].reset(new DataSamples(wordcount, maxbifeatid + 1));
		labels->Labels()[sampleidx].reset(new LabelVector(wordcount));

		samples->UnigramFeature()[sampleidx]->reserve(currunifeatcnts);
		samples->BigramFeature()[sampleidx]->reserve(currbifeatcnts);

		for (size_t pos = 0; pos < wordcount; ++pos) {

			for (int i = 0; i < currunifeatcnts[pos]; ++i) {
				samples->UnigramFeature()[sampleidx]->insert(pos, currunifeats[pos][i]) = 1;
			}

			for (int i = 0; i < currbifeatcnts[pos]; ++i) {
				samples->BigramFeature()[sampleidx]->insert(pos, currbifeats[pos][i]) = 1;
			}

			labels->Labels()[sampleidx]->coeffRef(pos) = currlabels[pos];
		}

		currunifeats.clear();
		currbifeats.clear();
		currunifeatcnts.clear();
		currbifeatcnts.clear();
		currlabels.clear();
	}
	return true;
}


bool build_vocab(const std::string& filepath, size_t cutoffvalue,
	boost::shared_ptr<Vocabulary>& words, boost::shared_ptr<Vocabulary>& labels){
	
	std::ofstream query, label;
	std::ifstream src(filepath);

	std::string queryfilepath = "queries.txt", labelfilepath = "labels.txt";
	query.open(queryfilepath);
	label.open(labelfilepath);

	if (!query.is_open()){
		BOOST_LOG_TRIVIAL(error) << "Failed to open query.txt" << std::endl;
		return false;
	}
	if (!label.is_open()){
		BOOST_LOG_TRIVIAL(error) << "Failed to open labels.txt" << std::endl;
		return false;
	}

	if (!src.is_open()){
		BOOST_LOG_TRIVIAL(error) << "Failed to open " << filepath << std::endl;
		return false;
	}

	src.getline(TempLineBuffer, sizeof(TempLineBuffer));
	char* ptr = nullptr;
	while (src.good()){		
		ptr = std::strtok(TempLineBuffer, "\t");
		if (ptr == nullptr){
			BOOST_LOG_TRIVIAL(fatal) << "File line format error" << TempLineBuffer << std::endl;
			return false;
		}
		query.write(ptr, std::strlen(ptr));
		query.write("\n", 1);

		ptr = std::strtok(NULL, "\t");
		if (ptr == nullptr){
			BOOST_LOG_TRIVIAL(fatal) << "File line format error " << TempLineBuffer << std::endl;
			return false;
		}
		label.write(ptr, std::strlen(ptr));
		label.write("\n", 1);

		ptr = std::strtok(NULL, "\t");
		if (ptr != nullptr){
			BOOST_LOG_TRIVIAL(fatal) << "Extra columns existed " << std::endl;
			return false;
		}
		src.getline(TempLineBuffer, sizeof(TempLineBuffer));
	}

	if (!src.eof()){
		BOOST_LOG_TRIVIAL(error) << "Unexpected error";
		return false;
	}

	src.close();
	query.close();
	label.close();

	words = Vocabulary::BuildVocabularyWithFilter(queryfilepath, cutoffvalue);
	labels = Vocabulary::BuildVocabulary(labelfilepath);
}

bool estimate_nn_sequence(const std::string& filepath, int& sparsebinary, int& sparsefloat, int& dense, int& label) {
	std::ifstream src(filepath);
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(error) << "Failed to open " << filepath;
		return false;
	}

	std::memset(TempLineBuffer, 0, sizeof(TempLineBuffer));
	src.getline(TempLineBuffer, sizeof(TempLineBuffer));
	char* ptr = nullptr;
	int linenumber = 0;
	while (src.good()) {
		if (std::strlen(TempLineBuffer) > 0) {
			ptr = std::strtok(TempLineBuffer, "\t");
		}
		src.getline(TempLineBuffer, sizeof(TempLineBuffer));
	}

	if (!src.eof()) {
		BOOST_LOG_TRIVIAL(error) << "Unexpected EOF";
		return false;
	}

	return true;
}

bool load_nn_sequence(const std::string& filepath,
	boost::shared_ptr<NNModel::NNSequenceFeature>& feats,
	boost::shared_ptr<NNModel::NNSequenceLabel>& labels){
	std::ifstream src(filepath);
	if (!src.is_open()){
		BOOST_LOG_TRIVIAL(error) << "Error to open file " << filepath;
		return false;
	}

	std::string line, seg;
	std::getline(src, line);
	std::vector<std::string> segs;
	while (src.good()){
		std::stringstream ss(line);
		std::getline(ss, seg, '\t');
		std::getline(src, line);
	}

	if (!src.eof()){
		BOOST_LOG_TRIVIAL(error) << "Unexpected EOF";
		return false;
	}


	return true;
}

template<>
bool DataLoader<kLibSVM, DataSamples, LabelVector>::LoadData() {
	if (filepath_.empty()) {
		valid_ = false;
	}
	else {
		valid_ = load_libsvm_data(filepath_, features_, labels_, !specifyfeatdim_, maxfeatid_ + 1);
		maxfeatid_ = features_->cols() - 1;
	}
	return valid_;
}

template<>
bool DataLoader<kLCCRF, LccrfSamples, LccrfLabels>::LoadData() {
	if (filepath_.empty()) {
		valid_ = false;
	}
	else {
		valid_ = load_lccrf_data_bin(filepath_, features_, labels_, !specifyfeatdim_,
			maxunifeatid_, maxbifeatid_, maxlabelid_);

		maxunifeatid_ = features_->GetMaxUnigramFeatureId();
		maxbifeatid_ = features_->GetMaxBigramFeatureId();
		maxlabelid_ = labels_->GetMaxLabelId();
	}
	return valid_;
}

template<>
bool DataLoader<kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::LoadData(){
	namespace fs = boost::filesystem;
	fs::path path(filepath_);
	if (!fs::exists(path) || !fs::is_regular_file(path)){
		BOOST_LOG_TRIVIAL(error) << "file path " << filepath_ << " bad";
		valid_ = false;
	}
	else {
		if (features_.get() == nullptr)
			features_.reset(new NNModel::NNSequenceFeature());
		if (labels_.get() == nullptr)
			labels_.reset(new NNModel::NNSequenceLabel());
	}


	return valid_;
}

template<>
bool DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>::LoadData(){
	namespace fs = boost::filesystem;
	fs::path path(filepath_);
	if (filepath_.empty() || !fs::exists(path)){
		valid_ = false;
		BOOST_LOG_TRIVIAL(info) << "either empty path or invaida path " << filepath_ << std::endl;
		return false;
	}
	else {

		if (features_.get() == nullptr){
			features_.reset(new NNModel::NNQueryFeature());
		}

		if (labels_.get() == nullptr){
			labels_.reset(new NNModel::NNQueryLabel());
		}
		
		if (!specifyfeatdim_){
			boost::shared_ptr<Vocabulary> words, labels;
			build_vocab(filepath_, cutoff_, words, labels);
			features_->SetVocabulary(words);
			labels_->SetVocabulary(labels);
		}

		NNModel::NNQueryFeaturizer featurizer(features_->GetVocabulary(), labels_->GetVocabulary());
		valid_ = featurizer.Featurize(features_, labels_, filepath_);
		return true;
	}
}

template<>
void DataLoader<kLibSVM, DataSamples, LabelVector>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kLibSVM, DataSamples, LabelVector>>& infosrc) {
	specifyfeatdim_ = true;
	maxfeatid_ = infosrc->maxfeatid_;
}

template<>
void DataLoader<kLCCRF, LccrfSamples, LccrfLabels>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kLCCRF, LccrfSamples, LccrfLabels>>& infosrc) {
	specifyfeatdim_ = true;
	maxunifeatid_ = infosrc->maxunifeatid_;
	maxbifeatid_ = infosrc->maxbifeatid_;
	maxlabelid_ = infosrc->maxlabelid_;
}

template<>
void DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>>& infosrc){
	specifyfeatdim_ = true;
	if (features_.get() == nullptr){
		features_.reset(new NNModel::NNQueryFeature());
	}

	if (labels_.get() == nullptr){
		labels_.reset(new NNModel::NNQueryLabel());
	}

	features_->SetVocabulary(infosrc->features_->GetVocabulary());
	labels_->SetVocabulary(infosrc->labels_->GetVocabulary());
}

template<>
void DataLoader<kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>>& infosrc){
	specifyfeatdim_ = true;

	if (features_.get() == nullptr)
		features_.reset(new NNModel::NNSequenceFeature());
	if (labels_.get() == nullptr)
		labels_.reset(new NNModel::NNSequenceLabel());

	features_->SetSparseBinarySize(infosrc->features_->GetSparseBinarySize());
	features_->SetSparseFloatSize(infosrc->features_->GetSparseFloatSize());
	features_->SetDenseSize(infosrc->features_->GetDenseSize());
	labels_->SetLabelSize(infosrc->labels_->GetLabelSize());
}

template class DataLoader<kLibSVM, DataSamples, LabelVector>;
template class DataLoader<kLCCRF, LccrfSamples, LccrfLabels>;
template class DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>;
template class DataLoader<kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>;