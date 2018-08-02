#include "dataop/ModelData.h"
#include "model/LccrfDataType.h"
#include "util/util.h"

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
const char *const
    ModelData<DataType, DataSampleType, DataLabelType>::kIoTrainPathOptions =
        "data.train";

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
const char *const
    ModelData<DataType, DataSampleType, DataLabelType>::kIoTestPathOptions =
        "data.test";

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
const char *const
    ModelData<DataType, DataSampleType, DataLabelType>::kIoModelPathOptions =
        "data.model";

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
const char
    *const ModelData<DataType, DataSampleType, DataLabelType>::kIoVocabOptions =
        "data.nncutoff";

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
ModelData<DataType, DataSampleType, DataLabelType>::ModelData()
    : iodesc_("Model data operations") {
  ConstructCmdOptions();
}

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
void ModelData<DataType, DataSampleType, DataLabelType>::InitFromCmd(
    int argc, const char *argv[]) {
  auto vm = ParseArgs(argc, argv, iodesc_, true);
  if (vm.count(kIoTrainPathOptions)) {
    trainpath_ = vm[kIoTrainPathOptions].template as<std::string>();
  } else {
    trainpath_.clear();
  }

  if (vm.count(kIoTestPathOptions)) {
    testpath_ = vm[kIoTestPathOptions].template as<std::string>();
  } else {
    testpath_.clear();
  }

  if (vm.count(kIoModelPathOptions)) {
    modelpath_ = vm[kIoModelPathOptions].template as<std::string>();
  } else {
    modelpath_.clear();
  }

  BOOST_LOG_TRIVIAL(info) << "Train data path : " << trainpath_;
  BOOST_LOG_TRIVIAL(info) << "Test data path  : " << testpath_;
  BOOST_LOG_TRIVIAL(info) << "Model file path : " << modelpath_;
  BOOST_LOG_TRIVIAL(info) << "NN Cutoff Value : " << nncutoff_;
}

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>>
ModelData<DataType, DataSampleType, DataLabelType>::RetrieveTrain() {
  boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> train;
  if (trainpath_.empty()) {
    return train;
  }
  train =
      boost::make_shared<DataLoader<DataType, DataSampleType, DataLabelType>>(
          trainpath_);
  train->SetCutoff(nncutoff_);
  return train;
}

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>>
ModelData<DataType, DataSampleType, DataLabelType>::RetrieveTest() {
  boost::shared_ptr<DataLoader<DataType, DataSampleType, DataLabelType>> test;
  if (testpath_.empty()) {
    return test;
  }
  test =
      boost::make_shared<DataLoader<DataType, DataSampleType, DataLabelType>>(
          testpath_);
  test->SetCutoff(nncutoff_);
  return test;
}

template <TrainDataType DataType, class DataSampleType, class DataLabelType>
void ModelData<DataType, DataSampleType, DataLabelType>::ConstructCmdOptions() {
  namespace po = boost::program_options;
  iodesc_.add_options()(kIoTrainPathOptions, po::value<std::string>(),
                        "train data path")(
      kIoTestPathOptions, po::value<std::string>(),
      "test data path")(kIoModelPathOptions, po::value<std::string>(),
                        "model file path for save or load")(
      kIoVocabOptions, po::value<size_t>(&nncutoff_)->default_value(3),
      "cutoff values when extract vocabulary");
}

template class ModelData<TrainDataType::kLibSVM, DataSamples, LabelVector>;
template class ModelData<TrainDataType::kLCCRF, LccrfSamples, LccrfLabels>;
template class ModelData<TrainDataType::kNNQuery, NNModel::NNQueryFeature,
                         NNModel::NNQueryLabel>;
template class ModelData<TrainDataType::kNNSequence, NNModel::NNSequenceFeature,
                         NNModel::NNSequenceLabel>;
