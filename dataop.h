#include <string>
#include <boost/shared_ptr.hpp>
#include "typedef.h"

#ifndef __DATA_OP_H__
#define __DATA_OP_H__

enum TrainDataType {
	kLibSVM,
	kLCCRF,
	kSemiCRF
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
	}

	bool LoadData();
	void SetMaxFeatureId(size_t featdim);

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

	void SetLccrfInfo(int unifeatid, int bifeatid, int labelid) {
		maxunifeatid_ = unifeatid;
		maxbifeatid_ = bifeatid;
		maxlabelid_ = labelid;
		specifyfeatdim_ = true;
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

	std::string filepath_;

	bool specifyfeatdim_;
	bool valid_;
};

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
