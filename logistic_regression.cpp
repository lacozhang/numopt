#include <iostream>
#include "dataop.h"
#include "DataIterator.h"
#include "cmdline.h"
#include "linearmodel.h"
#include "parameter.h"
#include "sgd.h"

int main(int argc, const char *argv[]) {

  Parameter param;
  bool ret = cmd_line_parse(argc, argv, param);
  if (!ret) {
	  return 0;
  }
  
  BOOST_LOG_TRIVIAL(info) << "load data";
  DataLoader<TrainDataType::kLibSVM, DataSamples, LabelVector> trainset(param.io_.train_);
  DataLoader<TrainDataType::kLibSVM, DataSamples, LabelVector> testset(param.io_.test_);

  trainset.LoadData();
  testset.SetMaxFeatureId(trainset.GetMaxFeatureId());
  testset.LoadData();
  
  DataIterator traindataiter(param.learn_.batchsize_, param.learn_.seed_);
  DataIterator testdataiter(param.learn_.batchsize_, param.learn_.seed_);
  
  BOOST_LOG_TRIVIAL(info) << "load finished";
  if (!trainset.IsValidDatset()) {
	  BOOST_LOG_TRIVIAL(fatal) << "load train set failed";
	  return 1;
  }
  else {
	  traindataiter.SetDataSet(trainset.GetData(), trainset.GetLabels());
  }

  if (testset.IsValidDatset()) {
	  testdataiter.SetDataSet(testset.GetData(), testset.GetLabels());
  }

  BinaryLinearModel model(param.loss_, trainset.GetMaxFeatureId() + 1, 1.0);

  switch (param.opt_) {
    case SGD:  // Stochastic Gradient Descent
    {
		StochasticGD sgd(param.learn_, traindataiter, testdataiter, model);
		sgd.Train();
		if (!param.io_.model_.empty()) {
			BOOST_LOG_TRIVIAL(trace) << "save model to file " << param.io_.model_;
			model.SaveModel(param.io_.model_);
		}
    } break;
    case PGD:  // Proximal Gradient Descent
    {
    } break;
    case GD:  // Gradient Descent
      break;
    case CG:  // Conjugate Gradient
      break;
    case LBFGS:  // Limited BFGS
      break;
    case CD:  // Coordinate Descent
      break;
    case BCD:  // Block Coordinate Descent
      break;
    default:
      break;
  }

  return 0;
}