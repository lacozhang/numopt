#define BOOST_TEST_MODULE "CRFIntegrationTest"
#include "dataop/DataIterator.h"
#include "model/LccrfDataType.h"
#include "model/LccrfModel.h"
#include "dataop/ModelData.h"
#include "crflib/crftemplate.h"
#include "crflib/lccrfeaturizer.h"
#include "util/util.h"
#include <boost/make_shared.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE(LccrfModelLoad) {
	ModelData<TrainDataType::kLCCRF, LccrfSamples, LccrfLabels> dat;
	const char* argv[] = { "test", "--data.train", "lccrf_test.bin", "--batch", "1" };
	int argc = 5;
	dat.InitFromCmd(argc, argv);
	auto train = dat.RetrieveTrain();
	train->LoadData();
	auto samples=train->GetData();
	auto label = train->GetLabels();
	
	DataIteratorBase<LccrfSamples, LccrfLabels> iterator;
	IndexData<LccrfSamples, LccrfLabels> indexdata(samples, label);

	iterator.InitFromCmd(argc, argv);
	iterator.SetDataSet(boost::make_shared<IndexData<LccrfSamples, LccrfLabels>>(samples, label));

	iterator.ResetBatch();

	LccrfSamples batchsample;
	LccrfLabels batchlabel, inferresult;

	LccrfModel model;
	model.InitFromData(iterator);

	model.GetParameters().setRandom();

	DenseVector densegrad;
	SparseVector sparsegrad;
	std::string summary;

	timeutil t;
	while (iterator.GetNextBatch(batchsample, batchlabel)) {

		t.tic();
		model.Learn(batchsample, batchlabel, densegrad);
		std::cout << "dense learn " << t.toc() << " ticks " << std::endl;

		t.tic();
		model.Learn(batchsample, batchlabel, sparsegrad);
		std::cout << "sparse learn " << t.toc() << " ticks " << std::endl;

		std::cout << "dense grad norm  " << densegrad.norm() << std::endl;
		std::cout << "sparse grad norm " << sparsegrad.norm() << std::endl;

		BOOST_ASSERT_MSG((densegrad - sparsegrad).norm() < 1e-3, "Error, difference between dense & sparse gradient");

		t.tic();
		model.Inference(batchsample, inferresult);
		std::cout << "inference " << t.toc() << " ticks " << std::endl;

		t.tic();
		model.Evaluate(batchsample, batchlabel, summary);
		std::cout << "evaluate " << t.toc() << " ticks " << std::endl;
		std::cout << summary << std::endl;
	}

	model.SaveModel("lccrf.test.model.bin");
	model.LoadModel("lccrf.test.model.bin");
}
