#define BOOST_TEST_MODULE "LccrfTester"
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/make_shared.hpp>
#include "../LccrfModel.h"
#include "../lccrf/crftemplate.h"
#include "../lccrf/lccrfeaturizer.h"
#include "../ModelData.h"
#include "../LccrfDataType.h"
#include "../DataIterator.h"
#include "../util.h"

BOOST_AUTO_TEST_CASE(LoadTemplates){

	CrfTemplate lccrf("template");
	std::vector<std::vector<std::string>> sentence{{"The", "DT", "B-NP"},
	{"Los", "NNP", "I-NP" },
	{"Angeles", "NNP", "I-NP"},
	{"Red", "NNP", "I-NP"},
	{"Cross", "NNP", "I-NP"},
	{"sent", "VBD", "B-VP"},
	{"2,480", "CD", "B-NP"},
	{"cots", "NNS", "I-NP"},
	{",", ",", "O"},
	{"500", "CD", "B-NP"},
	{"blankets", "NNS", "I-NP"},
	{",", ",", "O"},
	{"and", "CC", "O"},
	{"300", "CD", "B-NP"},
	{"pints", "NNS", "I-NP"},
	{"of", "IN", "B-PP" },
	{"Type-O", "JJ", "B-NP" },
	{"blood", "NN", "I-NP" },
	{".", ".", "O" } };

	std::vector<std::vector<std::string>> unifeat;
	std::vector<std::vector<std::string>> bifeat;

	lccrf.ExtractUnigramFeatures(sentence, unifeat);
	lccrf.ExtractBigramFeatures(sentence, bifeat);

	lccrf.SaveToFile("template.test.out.txt");
}

/*
BOOST_AUTO_TEST_CASE(LccrfFeaturizer) {
	LccrFeaturizer featurizer("template");
	featurizer.AccumulateFeatures("lccrf_test.txt", 3, 3);
	featurizer.FeaturizeFile("lccrf_test.txt", "lccrf_test.bin");
	featurizer.Save("lccrf");
	featurizer.Load("lccrf");
}
*/

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

		BOOST_ASSERT((densegrad - sparsegrad).norm() < 1e-3, "Error, difference between dense & sparse gradient");

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