#define BOOST_TEST_MODULE "DataOperationTest"
#include "dataop/dataop.h"
#include "util/util.h"
#include <boost/test/unit_test.hpp>
#include <fstream>

BOOST_AUTO_TEST_CASE(ReadNLines) {
  std::vector<std::string> lines;
  std::string inputfile = "D:\\zhangyu\\src\\data\\nn\\rnn-small\\train.txt";
  std::ifstream src(inputfile);
  while (read_lines(src, lines, 1000)) {
    LOG(INFO) << "Read " << lines.size() << " lines";
  }

  src.close();
  src.open(inputfile);
  while (read_sentence(src, lines)) {
    LOG(INFO) << "Sentence " << lines.size() << " lines";
  }
}

// TEST NN Sequence Loading
BOOST_AUTO_TEST_CASE(LoadNNSequence) {
  std::string inputfile = "D:\\zhangyu\\src\\data\\nn\\rnn\\train.txt";
  std::string inputfile2 =
      "D:\\zhangyu\\src\\data\\nn\\rnn-small\\small.test.txt";
  DataLoader<TrainDataType::kNNSequence, NNModel::NNSequenceFeature,
             NNModel::NNSequenceLabel>
      loader(inputfile);
  loader.LoadData();
}

// Test NN Query Loading
BOOST_AUTO_TEST_CASE(LoadNNQuery) {
  std::string inputfile = "D:\\zhangyu\\src\\data\\nn\\cnn\\huge.train.txt";
  DataLoader<TrainDataType::kNNQuery, NNModel::NNQueryFeature,
             NNModel::NNQueryLabel>
      loader(inputfile);
  loader.SetCutoff(2);
  timeutil timer;
  timer.tic();
  loader.LoadData();
  LOG(INFO) << "Cost " << timer.toc() << " seconds";
}
