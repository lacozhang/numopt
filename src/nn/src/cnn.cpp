#include "nn/cnn.h"
#include "util/util.h"

namespace NNModel {
const char *CNNModel::kEmbeddingSizeOption = "cnn.embed";
const char *CNNModel::kHiddenSizeOption = "cnn.hidden";
const char *CNNModel::kConvFilterSizeOptioin = "cnn.convcnt";
const char *CNNModel::kConvWindowSizeOption = "cnn.convsize";
const char *CNNModel::kPoolingSizeOption = "cnn.pool";
const char *CNNModel::kDropOutOption = "cnn.dropout";

CNNModel::CNNModel() {
  optionsdesc_.add_options()(
      kEmbeddingSizeOption,
      boost::program_options::value<int>(&embedsize_)->default_value(50),
      "embedding size")(
      kHiddenSizeOption,
      boost::program_options::value<int>(&hiddensize_)->default_value(100),
      "hidden size of cnn")(
      kConvFilterSizeOptioin,
      boost::program_options::value<int>(&convfilters_)->default_value(50),
      "number of filters of cnn")(
      kConvWindowSizeOption,
      boost::program_options::value<int>(&convsize_)->default_value(3),
      "number of words for convolution")(
      kPoolingSizeOption,
      boost::program_options::value<int>(&poolstack_)->default_value(3),
      "max strides for convolution operation")(
      kDropOutOption,
      boost::program_options::value<double>(&dropout_)->default_value(0.5),
      "drop out");

  vocabsize_ = labelsize_ = 0;
}

CNNModel::~CNNModel() {}

void CNNModel::InitFromCmd(int argc, const char *argv[]) {
  auto vm = ParseArgs(argc, argv, optionsdesc_, true);
  LOG(INFO) << "Embedding Size " << embedsize_;
  LOG(INFO) << "Hidden Size    " << hiddensize_;
  LOG(INFO) << "#Filters       " << convfilters_;
  LOG(INFO) << "#Window Size   " << convsize_;
  LOG(INFO) << "#Stack Size    " << poolstack_;
  LOG(INFO) << "DropOut        " << dropout_;
}

void CNNModel::InitFromData(DataIterator &iterator) {
  auto data = iterator.GetAllData();
  auto label = iterator.GetAllLabel();
  vocab_ = data.GetVocabulary();
  vocabsize_ = data.GetVocabularySize();
  label_ = label.GetVocabulary();
  labelsize_ = label.GetLabelSize();
}

void CNNModel::Init() {
  int modelsize = 0;
  // aligned memory by 16 bytes
  // embedding size
  modelsize += vocabsize_ * embedsize_;
  // convolution size
  modelsize += convfilters_ * embedsize_ * convsize_;
  // from pooling to hidden
  modelsize += poolstack_ * embedsize_ * hiddensize_;
  // from query to hidden
  modelsize += hiddensize_ * vocabsize_;
  // from hidden to output
  modelsize += hiddensize_ * labelsize_;
  // from query to output
  modelsize += labelsize_ * vocabsize_;

  param_ = boost::make_shared<RealVector>(modelsize);
  grad_ = boost::make_shared<RealVector>(modelsize);
  if (!param_ || !grad_) {
    LOG(ERROR) << "Allocate memory failed";
    std::abort();
  }
  param_->setZero();
  grad_->setZero();

  double *parambae = param_->data(), *gradbase = grad_->data();
  embedlayer_ = boost::make_shared<EmbeddingLayer>(parambae, gradbase,
                                                   vocabsize_, embedsize_);
  parambae += vocabsize_ * embedsize_;
  gradbase += vocabsize_ * embedsize_;
  convlayer_ = boost::make_shared<TextConvLayer>(
      parambae, gradbase, convfilters_, convsize_, embedsize_, 1);
  poolinglayer_ =
      boost::make_shared<TextMaxPoolingLayer>(poolstack_, embedsize_);
  poolactlayer_ = boost::make_shared<ActivateLayer>(ActivateType::Sigmoid);
  parambae += convfilters_ * convsize_ * embedsize_;
  gradbase += convfilters_ * convsize_ * embedsize_;
  hiddenlayer_ = boost::make_shared<LinearLayer>(
      parambae, gradbase, poolstack_ * embedsize_, hiddensize_);
  parambae += poolstack_ * embedsize_ * hiddensize_;
  gradbase += poolstack_ * embedsize_ * hiddensize_;
  bowhiddenlayer_ = boost::make_shared<EmbeddingSumLayer>(
      parambae, gradbase, vocabsize_, hiddensize_);
  parambae += vocabsize_ * hiddensize_;
  gradbase += vocabsize_ * hiddensize_;

  hiddensumlayer_ = boost::make_shared<VectorSumLayer>(2, hiddensize_);
  hiddenactlayer_ = boost::make_shared<ActivateLayer>(ActivateType::Sigmoid);

  hiddendroplayer_ = boost::make_shared<DropoutLayer>(dropout_, hiddensize_);

  outputlayer_ = boost::make_shared<LinearLayer>(parambae, gradbase,
                                                 hiddensize_, labelsize_);
  parambae += hiddensize_ * labelsize_;
  gradbase += hiddensize_ * labelsize_;

  bowoutputlayer_ = boost::make_shared<EmbeddingSumLayer>(
      parambae, gradbase, vocabsize_, labelsize_);
  outputsumlayer_ = boost::make_shared<VectorSumLayer>(2, labelsize_);

  losslayer_ = boost::make_shared<CrossEntropyLoss>(labelsize_);
}

} // namespace NNModel
