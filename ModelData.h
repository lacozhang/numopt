#pragma once
#include <boost/program_options.hpp>
#include "dataop.h"

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
class ModelData {
public:
	ModelData();
	virtual ~ModelData() {

	}

	void InitFromCmd(int argc, const char* argv[]);
	boost::program_options::options_description Options() {
		return iodesc_;
	}
	boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> RetrieveTrain();
	boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> RetrieveTest();
	std::string ModelFilePath() {
		return modelpath_;
	}

private:
	void ConstructCmdOptions();
	boost::program_options::options_description iodesc_;

	std::string trainpath_;
	std::string testpath_;
	std::string modelpath_;

	static const char* const kIoTrainPathOptions;
	static const char* const kIoTestPathOptions;
	static const char* const kIoModelPathOptions;
};
