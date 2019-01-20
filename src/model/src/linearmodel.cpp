#include "model/linearmodel.h"
#include "util/cmdline.h"
#include "util/util.h"
#include <boost/algorithm/string.hpp>
#include <boost/make_shared.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

const char *BinaryLinearModel::kLossOption = "linear.loss";
const char *BinaryLinearModel::kBiasOption = "linear.bias";

BinaryLinearModel::BinaryLinearModel() {
  optionsdesc_.add_options()(
      BinaryLinearModel::kLossOption,
      boost::program_options::value<std::string>()->default_value("logistic"),
      "loss function: squared/hinge/logistic/squaredhinge\nsquared: used for "
      "regression.\nhinge,logistic,squaredhinge: used for classification")(
      BinaryLinearModel::kBiasOption,
      boost::program_options::value<double>()->default_value(0),
      "bias for linear model");
}

void BinaryLinearModel::SetLoss(LossFunc loss) {
  switch (loss) {
  case Squared:
    loss_ = boost::make_shared<SquaredLoss>();
    break;
  case Hinge:
    loss_ = boost::make_shared<HingeLoss>();
    break;
  case Logistic:
    loss_ = boost::make_shared<LogLoss>();
    break;
  case SquaredHinge:
    loss_ = boost::make_shared<SquaredHingeLoss>();
    break;
  default:
    LOG(FATAL) << "Loss function error";
    break;
  }
  this->losstype_ = loss;
}

void BinaryLinearModel::InitFromCmd(int argc, const char *argv[]) {
  auto vm = ParseArgs(argc, argv, optionsdesc_, true);
  try {
    std::string lossoption =
        vm[BinaryLinearModel::kLossOption].as<std::string>();
    LOG(INFO) << "Loss Function : " << lossoption;
    SetLoss(parselossfunc(
        vm[BinaryLinearModel::kLossOption].as<std::string>().c_str()));
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    throw e;
  }

  try {
    bias_ = vm[BinaryLinearModel::kBiasOption].as<double>();
    LOG(INFO) << "Bias           : " << bias_;
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    throw e;
  }
}

void BinaryLinearModel::InitFromData(DataIterator &iterator) {
  auto dat = iterator.GetAllData();
  featdim_ = dat.cols();
  param_.reset(new DenseVector(featdim_));
  LOG(INFO) << "param dimension " << featdim_;
  if (param_.get() == nullptr) {
    LOG(ERROR)
        << "Allocate parameter vector for binary model failed";
  } else {
    param_->setZero();
  }
}

BinaryLinearModel::~BinaryLinearModel() {}

bool BinaryLinearModel::LoadModel(std::string model) {
  bool binfmode = boost::algorithm::ends_with(model, ".bin");
  std::ifstream src(model.c_str(),
                    binfmode ? std::ios_base::binary | std::ios_base::in
                             : std::ios_base::in);
  if (!src.is_open()) {
    LOG(ERROR) << "open file " << model << " failed";
    return false;
  }
  src.read((char *)&featdim_, sizeof(size_t));
  for (size_t i = 0; i < featdim_; ++i) {
    src.read((char *)&param_->coeffRef(i),
             sizeof(DenseVector::CoeffReturnType));
  }
  return true;
}

bool BinaryLinearModel::SaveModel(std::string model) {
  bool filemode = boost::algorithm::ends_with(model, ".bin");
  std::ofstream writer(model.c_str(),
                       filemode ? std::ios_base::binary | std::ios_base::out
                                : std::ios_base::out);
  if (!writer.is_open()) {
    LOG(ERROR) << "open file " << model << " failed";
    return false;
  }
  writer.write((char *)&featdim_, sizeof(size_t));
  for (size_t i = 0; i < featdim_; ++i) {
    writer.write((char *)&param_->coeff(i),
                 sizeof(DenseVector::CoeffReturnType));
  }
  return true;
}

double BinaryLinearModel::Learn(DataSamples &samples, LabelVector &labels,
                                SparseVector &grad) {
  double avgloss = 0.0;
  grad.resize(featdim_);
  DenseVector xtw = samples * (*param_);
  grad.resizeNonZeros(samples.nonZeros());
  grad.setZero();

  for (int i = 0; i < samples.rows(); ++i) {
    double loss = loss_->dloss(xtw.coeff(i), labels.coeff(i));
    avgloss += loss_->loss(xtw.coeff(i), labels.coeff(i));
    for (DataSamples::InnerIterator it(samples, i); it; ++it) {
      grad.coeffRef(it.col()) += loss * it.value();
    }
  }
  grad /= samples.rows();
  avgloss /= samples.rows();
  return avgloss;
}

double BinaryLinearModel::Learn(DataSamples &samples, LabelVector &labels,
                                DenseVector &grad) {
  double avgloss = 0;
  DenseVector xtw = samples * (*param_);
  grad.resize(featdim_);
  grad.setZero();

  for (int i = 0; i < labels.rows(); ++i) {
    grad += samples.row(i) * loss_->dloss(xtw.coeff(i), labels.coeff(i));
    avgloss += loss_->loss(xtw.coeff(i), labels.coeff(i));
  }
  grad /= samples.rows();
  avgloss /= samples.rows();
  return avgloss;
}

void BinaryLinearModel::Inference(DataSamples &samples, LabelVector &labels) {
  DenseVector xtw = samples * (*param_);
  labels.resize(xtw.rows());
  for (int i = 0; i < labels.rows(); ++i) {
    labels.coeffRef(i) = xtw.coeff(i) > 0 ? 1 : -1;
  }
}

double BinaryLinearModel::Evaluate(DataSamples &samples, LabelVector &labels,
                                   std::string &summary) {
  std::stringstream sout;
  DenseVector xtw = samples * (*param_);
  double losses = 0.0f;
  double tp = 0.0f, fp = 0.0f, tn = 0.0f, fn = 0.0f;
  double correct = 0.0f;
  for (int i = 0; i < labels.rows(); ++i) {
    losses += loss_->loss(xtw.coeff(i), labels.coeff(i));
    if (xtw.coeff(i) * labels.coeff(i) > 0) {
      correct += 1.0;
      if (labels.coeff(i) > 0) {
        tp += 1.0;
      } else {
        tn += 1.0;
      }
    } else {
      if (labels.coeff(i) > 0) {
        fn += 1.0;
      } else {
        fp += 1.0;
      }
    }
  }

  losses /= samples.rows();
  double precision = 0.0, recall = 0.0f;
  if ((tp + fp) > 0) {
    precision = tp / (tp + fp);
  }

  if ((tp + fn) > 0) {
    recall = tp / (tp + fn);
  }

  sout << "avg loss: " << losses << "/precision: " << precision
       << "/recall: " << recall << "/accu: " << (correct / samples.rows());
  summary = sout.str();
  return losses;
}
