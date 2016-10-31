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
	DataLoader(std::string);
	bool LoadData();
	void SetFeatureDimension(size_t featdim);

private:

	boost::shared_ptr<Feat> features_;
	boost::shared_ptr<Label> labels_;

	size_t featdim_;
	bool specifyfeatdim_;
	std::string filepath_;
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
