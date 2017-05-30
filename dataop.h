#include <string>
#include <boost/shared_ptr.hpp>
#include "typedef.h"
#include "LccrfDataType.h"
#include "nn/nnquery.h"
#include "nn/nnsequencedata.h"

#ifndef __DATA_OP_H__
#define __DATA_OP_H__

enum TrainDataType {
	kLibSVM,
	kLCCRF,
	kSemiCRF,
	kNNQuery,
	kNNSequence
};

template <TrainDataType T, class Feat, class Label>
class DataLoader {

public:
	DataLoader(std::string srcfile) {
		filepath_ = srcfile;
		specifyfeatdim_ = false;
		maxfeatid_ = 0;
		valid_ = false;

		// init for lccrf
		maxunifeatid_ = 0;
		maxbifeatid_ = 0;
		maxlabelid_ = 0;

		cutoff_ = 0;
	}

	bool LoadData();

	size_t MaxFeatureId() {
		return maxfeatid_;
	}

	void SetCutoff(size_t v){
		cutoff_ = v;
	}

	void SetModelMetaInfo(const boost::shared_ptr<DataLoader<T, Feat, Label>>& infosrc);

	size_t GetMaxFeatureId() const {
		return maxfeatid_;
	}

	// for lccrf
	int GetMaxUnigramFeatureId() const {
		return maxunifeatid_;
	}

	int GetMaxBigramFeatureId() const {
		return maxbifeatid_;
	}

	int GetMaxLabelId() const {
		return maxlabelid_;
	}

	// for all
	boost::shared_ptr<Feat>& GetData() {
		return features_;
	}

	boost::shared_ptr<Label>& GetLabels() {
		return labels_;
	}

	bool IsValidDatset() const {
		return valid_;
	}

private:

	// for all models
	boost::shared_ptr<Feat> features_;
	boost::shared_ptr<Label> labels_;

	// for svm model
	size_t maxfeatid_;

	// for lccrf
	int maxunifeatid_;
	int maxbifeatid_;
	int maxlabelid_;

	// for nn
	size_t cutoff_;

	std::string filepath_;

	bool specifyfeatdim_;
	bool valid_;
};

template<>
bool DataLoader<kLibSVM, DataSamples, LabelVector>::LoadData();

template<>
bool DataLoader<kLCCRF, LccrfSamples, LccrfLabels>::LoadData();

template<>
bool DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>::LoadData();

template<>
bool DataLoader<kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::LoadData();

template<>
void DataLoader<kLibSVM, DataSamples, LabelVector>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kLibSVM, DataSamples, LabelVector>>& infosrc);

template<>
void DataLoader<kLCCRF, LccrfSamples, LccrfLabels>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kLCCRF, LccrfSamples, LccrfLabels>>& infosrc);

template<>
void DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>>& infosrc);

template<>
void DataLoader<kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>::SetModelMetaInfo(
	const boost::shared_ptr<DataLoader<kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel>>& infosrc);

bool matrix_size_estimation(std::string featfile, Eigen::VectorXi &datsize,
                            int &row, int &col);

bool load_libsvm_data(std::string featfile,
                      boost::shared_ptr<DataSamples> &Samples,
                      boost::shared_ptr<LabelVector> &labels, bool estimate,
                      int colsize);

bool save_libsvm_data_bin(std::string filepath,
	boost::shared_ptr<DataSamples>& samples,
	boost::shared_ptr<LabelVector>& labels);

void parselibsvmline(char *line, std::vector<std::pair<size_t, float>> &feats,
	int &label, bool parse = true);


#endif
