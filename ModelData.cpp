#include "ModelData.h"
#include "util.h"

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
const char* const ModelData<DataType, DataSampleType, DataLabelType>::kIoTrainPathOptions = "data.train";

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
const char* const ModelData<DataType, DataSampleType, DataLabelType>::kIoTestPathOptions = "data.test";

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
const char* const ModelData<DataType, DataSampleType, DataLabelType>::kIoModelPathOptions = "data.model";

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
ModelData<DataType, DataSampleType, DataLabelType>::ModelData()
{
	ConstructCmdOptions();
}

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
void ModelData<DataType, DataSampleType, DataLabelType>::InitFromCmd(int argc, const char * argv[])
{
	auto vm = ParseArgs(argc, argv, iodesc_, true);
	if (vm.count(kIoTrainPathOptions)) {
		trainpath_ = vm[kIoTrainPathOptions].as<std::string>();
	}
	else {
		trainpath_.clear();
	}

	if (vm.count(kIoTestPathOptions)) {
		testpath_ = vm[kIoTestPathOptions].as<std::string>();
	}
	else {
		testpath_.clear();
	}

	if (vm.count(kIoModelPathOptions)) {
		modelpath_ = vm[kIoModelPathOptions].as<std::string>();
	}
	else {
		modelpath_.clear();
	}

	BOOST_LOG_TRIVIAL(info) << "Train data path : " << trainpath_;
	BOOST_LOG_TRIVIAL(info) << "Test data path  : " << testpath_;
	BOOST_LOG_TRIVIAL(info) << "Model file path : " << modelpath_;
}

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> ModelData<DataType, DataSampleType, DataLabelType>::RetrieveTrain()
{
	boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> train;
	if (trainpath_.empty()) {
		return train;
	}
	train.reset(new DataLoader<DataType, DataSamples, DataLabelType>(trainpath_));
	return train;
}

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> ModelData<DataType, DataSampleType, DataLabelType>::RetrieveTest()
{
	boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> test;
	if (testpath_.empty()) {
		return test;
	}
	test.reset(new DataLoader<DataType, DataSampleType, DataLabelType>(testpath_));
	return test;
}

template<TrainDataType DataType, class DataSampleType, class DataLabelType>
void ModelData<DataType, DataSampleType, DataLabelType>::ConstructCmdOptions()
{
	namespace po = boost::program_options;
	iodesc_.add_options()
		(kIoTrainPathOptions, po::value<std::string>(), "train data path")
		(kIoTestPathOptions, po::value<std::string>(), "test data path")
		(kIoModelPathOptions, po::value<std::string>(), "model file path for save or load");
}

template class ModelData<TrainDataType::kLibSVM, DataSamples, LabelVector>;