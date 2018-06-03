#pragma once
#include "dataop/DataIterator.h"
#include "dataop/ModelData.h"
#include "dataop/dataop.h"
#include "model/LccrfModel.h"
#include "model/linearmodel.h"
#include "optimizer/optimfactory.h"
#include "util/cmdline.h"
#include "util/parameter.h"
#include <boost/make_shared.hpp>

template <class ParameterType, class DataSampleType, class DataLabelType,
          class SparseGradientType, class DenseGradientType>
boost::shared_ptr<OptMethodBase<ParameterType, DataSampleType, DataLabelType,
                                SparseGradientType, DenseGradientType>>
CreateOptimizer(
    std::string optimname,
    boost::shared_ptr<
        AbstractModel<ParameterType, DataSampleType, DataLabelType,
                      SparseGradientType, DenseGradientType>>
        model,
    BaseOptimizerFactory<ParameterType, DataSampleType, DataLabelType,
                         SparseGradientType, DenseGradientType> &optimfactory) {

  boost::shared_ptr<OptMethodBase<ParameterType, DataSampleType, DataLabelType,
                                  SparseGradientType, DenseGradientType>>
      optimizer;
  optimizer.reset(optimfactory.Create(optimname, *model));
  return optimizer;
}

template <TrainDataType DataType, class ParameterType, class DataSampleType,
          class DataLabelType, class SparseGradientType,
          class DenseGradientType>
void RunModel(
    int argc, const char *argv[], std::string optimizertype,
    boost::shared_ptr<
        AbstractModel<ParameterType, DataSampleType, DataLabelType,
                      SparseGradientType, DenseGradientType>>
        model,
    BaseOptimizerFactory<ParameterType, DataSampleType, DataLabelType,
                         SparseGradientType, DenseGradientType> &optimfactory) {

  BOOST_LOG_TRIVIAL(info) << "load data";

  ModelData<DataType, DataSampleType, DataLabelType> modeldata;
  modeldata.InitFromCmd(argc, argv);

  auto trainset = modeldata.RetrieveTrain();
  auto testset = modeldata.RetrieveTest();

  if (trainset == nullptr) {
    BOOST_LOG_TRIVIAL(fatal) << "Train set not set, failed";
    return;
  }

  trainset->LoadData();
  if (testset != nullptr) {
    testset->SetModelMetaInfo(trainset);
    testset->LoadData();
  }

  DataIteratorBase<DataSampleType, DataLabelType> trainiter, testiter;
  trainiter.InitFromCmd(argc, argv);
  trainiter.SetDataSet(
      boost::make_shared<IndexData<DataSampleType, DataLabelType>>(
          trainset->GetData(), trainset->GetLabels()));

  if (testset != nullptr) {
    testiter.SetDataSet(
        boost::make_shared<IndexData<DataSampleType, DataLabelType>>(
            testset->GetData(), testset->GetLabels()));
  }

  model->InitFromCmd(argc, argv);
  model->InitFromData(trainiter);
  model->Init();

  boost::shared_ptr<OptMethodBase<ParameterType, DataSampleType, DataLabelType,
                                  SparseGradientType, DenseGradientType>>
      optimizer;
  optimizer = CreateOptimizer<ParameterType, DataSampleType, DataLabelType,
                              SparseGradientType, DenseGradientType>(
      optimizertype, model, optimfactory);
  optimizer->InitFromCmd(argc, argv);
  optimizer->SetTrainData(
      boost::make_shared<DataIteratorBase<DataSampleType, DataLabelType>>(
          trainiter));
  optimizer->SetTestData(
      boost::make_shared<DataIteratorBase<DataSampleType, DataLabelType>>(
          testiter));
  optimizer->Train();
  if (!modeldata.ModelFilePath().empty()) {
    BOOST_LOG_TRIVIAL(trace)
        << "save model to file " << modeldata.ModelFilePath();
    model->SaveModel(modeldata.ModelFilePath());
  }
}

template <TrainDataType DataType, class ParameterType, class DataSampleType,
          class DataLabelType, class SparseGradientType,
          class DenseGradientType>
void RunModelHelp(
    std::string optimtype,
    boost::shared_ptr<
        AbstractModel<ParameterType, DataSampleType, DataLabelType,
                      SparseGradientType, DenseGradientType>>
        model,
    BaseOptimizerFactory<ParameterType, DataSampleType, DataLabelType,
                         SparseGradientType, DenseGradientType> &optimfactory) {

  std::cout << model->Options();
  auto optimizer = CreateOptimizer<ParameterType, DataSampleType, DataLabelType,
                                   SparseGradientType, DenseGradientType>(
      optimtype, model, optimfactory);
  std::cout << optimizer->Options();
  ModelData<DataType, DataSampleType, DataLabelType> modeldata;
  std::cout << modeldata.Options();
  DataIteratorBase<DataSampleType, DataLabelType> iterobject;
  std::cout << iterobject.Options();
}
