#pragma once
#include <boost/make_shared.hpp>
#include "dataop.h"
#include "ModelData.h"
#include "DataIterator.h"
#include "cmdline.h"
#include "linearmodel.h"
#include "LccrfModel.h"
#include "parameter.h"
#include "optimizer/sgd.h"


template<class ParameterType, class DataSampleType, class DataLabelType, class SparseGradientType, class DenseGradientType>
boost::shared_ptr<OptMethodBase<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>> CreateOptimizer(OptMethod optimtype,
	boost::shared_ptr<AbstractModel<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>> model) {

	boost::shared_ptr<OptMethodBase<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>> optimizer;
	switch (optimtype) {
	case OptMethod::SGD:  // Stochastic Gradient Descent
	{
		optimizer.reset(new  StochasticGD<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>(*model));
	}
	break;
	case OptMethod::PGD:  // Proximal Gradient Descent
		break;
	case OptMethod::GD:  // Gradient Descent
		break;
	case OptMethod::CG:  // Conjugate Gradient
		break;
	case OptMethod::LBFGS:  // Limited BFGS
		break;
	case OptMethod::CD:  // Coordinate Descent
		break;
	case OptMethod::BCD:  // Block Coordinate Descent
		break;
	case OptMethod::None:
	{

	}
		break;
	}

	return optimizer;
}

template<TrainDataType DataType, class ParameterType, class DataSampleType, class DataLabelType,class SparseGradientType,class DenseGradientType>
void RunModel(int argc, const char* argv[], OptMethod optimizertype, boost::shared_ptr<AbstractModel<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>> model) {

	BOOST_LOG_TRIVIAL(info) << "load data";

	ModelData<DataType, DataSampleType, DataLabelType> modeldata;
	modeldata.InitFromCmd(argc, argv);

	auto trainset = modeldata.RetrieveTrain();
	auto testset = modeldata.RetrieveTest();

	if (trainset == nullptr){
		BOOST_LOG_TRIVIAL(fatal) << "Train set not set, failed";
		return;
	}

	trainset->LoadData();
	if (testset != nullptr){
		testset->SetModelMetaInfo(trainset);
		testset->LoadData();
	}

	DataIteratorBase<DataSampleType, DataLabelType> trainiter, testiter;
	trainiter.InitFromCmd(argc, argv);
	trainiter.SetDataSet(boost::make_shared<IndexData<DataSampleType, DataLabelType>>(trainset->GetData(), trainset->GetLabels()));

	if (testset != nullptr){
		testiter.SetDataSet(boost::make_shared<IndexData<DataSampleType, DataLabelType>>(testset->GetData(), testset->GetLabels()));
	}

	model->InitFromCmd(argc, argv);
	model->InitFromData(trainiter);

	boost::shared_ptr<OptMethodBase<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>> optimizer;
	optimizer = CreateOptimizer<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>(optimizertype, model);
	optimizer->InitFromCmd(argc, argv);
	optimizer->SetTrainData(boost::make_shared<DataIteratorBase<DataSampleType, DataLabelType>>(trainiter));
	optimizer->SetTestData(boost::make_shared<DataIteratorBase<DataSampleType, DataLabelType>>(testiter));
	optimizer->Train();
	if (!modeldata.ModelFilePath().empty()) {
		BOOST_LOG_TRIVIAL(trace) << "save model to file " << modeldata.ModelFilePath();
		model->SaveModel(modeldata.ModelFilePath());
	}
}

template<TrainDataType DataType, class ParameterType, class DataSampleType, class DataLabelType, class SparseGradientType, class DenseGradientType>
void RunModelHelp(OptMethod optimtype, boost::shared_ptr<AbstractModel<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>> model) {

	std::cout << model->Options();
	auto optimizer = CreateOptimizer<ParameterType, DataSampleType, DataLabelType, SparseGradientType, DenseGradientType>(optimtype,model);
	std::cout << optimizer->Options();
	ModelData<DataType, DataSampleType, DataLabelType> modeldata;
	std::cout << modeldata.Options();
	DataIteratorBase<DataSampleType, DataLabelType> iterobject;
	std::cout << iterobject.Options();
}