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


BOOST_AUTO_TEST_CASE(ReadNLines) {
    std::vector<std::string> lines;
    std::string inputfile = "D:\\zhangyu\\src\\data\\nn\\rnn-small\\train.txt";
    std::ifstream src(inputfile);
    while (read_lines(src, lines, 1000)) {
        BOOST_LOG_TRIVIAL(info) << "Read " << lines.size() << " lines";
    }

    src.close();
    src.open(inputfile);
    while (read_sentence(src, lines)) {
        BOOST_LOG_TRIVIAL(info) << "Sentence " << lines.size() << " lines";
    }
}

// TEST NN Sequence Loading
BOOST_AUTO_TEST_CASE(LoadNNSequence) {
    std::string inputfile = "D:\\zhangyu\\src\\data\\nn\\rnn\\train.txt";
    std::string inputfile2 = "D:\\zhangyu\\src\\data\\nn\\rnn-small\\small.test.txt";
    DataLoader<TrainDataType::kNNSequence, NNModel::NNSequenceFeature, NNModel::NNSequenceLabel> loader(inputfile);
    loader.LoadData();
}

// Test NN Query Loading
BOOST_AUTO_TEST_CASE(LoadNNQuery) {
    std::string inputfile = "D:\\zhangyu\\src\\data\\nn\\cnn\\huge.train.txt";
    DataLoader<TrainDataType::kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel> loader(inputfile);
    loader.SetCutoff(2);
    timeutil timer;
    timer.tic();
    loader.LoadData();
    BOOST_LOG_TRIVIAL(info) << "Cost " << timer.toc() << " seconds";
}

// Test Split Function
BOOST_AUTO_TEST_CASE(StringSplitOperation) {
	std::string test1 = "i'm going to find a test  case  u";
	std::vector<std::string> res;
	Util::Split(test1, res, " ", true);
	res.clear();
	Util::Split(test1, res, " ", false);
	res.clear();

	std::string test2 = "c \tb u\t\t\td";
	Util::Split(test2, res, "\t", false);
	res.clear();
	Util::Split(test2, res, "\t", true);
	res.clear();

	std::string test3 = "ax dy x\t\t\t";
	Util::Split(test3, res, "\t", false);
	res.clear();

	std::string test4 = "abcd e d xe";
	Util::Split(test4, res, "\t", false);

}


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