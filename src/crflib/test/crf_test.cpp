#pragma once
#define BOOST_TEST_MODULE "CRFTest"
#include <boost/test/unit_test.hpp>
#include <string>


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