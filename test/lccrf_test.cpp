#define BOOST_TEST_MODULE "LccrfTester"

#include <boost/test/unit_test.hpp>
#include "../lccrf/crftemplate.h"
#include "../lccrf/lccrfeaturizer.h"
#include "../ModelData.h"
#include "../LccrfDataType.h"

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


BOOST_AUTO_TEST_CASE(LccrfFeaturizer) {
	LccrFeaturizer featurizer("template");
	featurizer.AccumulateFeatures("lccrf_test.txt", 3, 3);
	featurizer.FeaturizeFile("lccrf_test.txt", "lccrf_test.bin");
	featurizer.Save("lccrf");
	featurizer.Load("lccrf");
}

BOOST_AUTO_TEST_CASE(LccrfModelLoad) {
	ModelData<TrainDataType::kLCCRF, LccrfSamples, LccrfLabels> dat;
	const char* argv[] = { "test", "--data.train", "lccrf_test.bin" };
	dat.InitFromCmd(3, argv);
	auto train = dat.RetrieveTrain();
	train->LoadData();
	auto samples=train->GetData();
	auto label = train->GetLabels();
}