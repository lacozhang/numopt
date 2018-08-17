/*
 * =====================================================================================
 *
 *       Filename:  stream_reader_test.cpp
 *
 *    Description:  Test for stream reader
 *
 *        Version:  1.0
 *        Created:  07/08/2018 21:30:54
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "data/stream_reader.h"
#include "gtest/gtest.h"
using namespace mltools;
using namespace std;
namespace {
string libsvmPath = "/Users/edwinzhang/src/parameter_server/example/linear/"
                    "data/ctr/train/part-00[1-4].gz";
string relLibsvmPath =
    "../../parameter_server/example/linear/data/ctr/train/part-00[1-4].gz";

string rcvDataPath = "/Users/edwinzhang/src/parameter_server/example/linear/"
                     "data/rcv1/train/part-00[1-4]";
} // namespace

namespace mltools {
DEFINE_int32(num_threads, 2, "threads of numbers");
}

TEST(SR, init) { mltools::StreamReader<float> test; }

TEST(SR, searchFile) {
  DataConfig cfg;
  cfg.set_format(DataConfig::TEXT);
  cfg.add_file(libsvmPath);
  cout << cfg.DebugString();
  auto res = searchFiles(cfg);
  CHECK_EQ(res.file_size(), 4);
  cout << res.DebugString();

  cfg.clear_file();
  cfg.add_file(relLibsvmPath);
  cout << "config format " << endl;
  cout << cfg.DebugString();
  res = searchFiles(cfg);
  CHECK_EQ(res.file_size(), 4);
  cout << res.DebugString();
}

TEST(SR, readExamples) {
  DataConfig cfg;
  cfg.set_format(DataConfig::TEXT);
  cfg.set_text(DataConfig::LIBSVM);
  cfg.add_file(rcvDataPath);
  auto res = searchFiles(cfg);
  StreamReader<float> reader(res);
  vector<Example> examples;
  int size = 20000;
  int totalCount = 0;
  //  while (reader.readExamples(size, &examples)) {
  //    LOG(INFO) << "read size " << examples.size();
  //    totalCount += examples.size();
  //  }
  LOG(INFO) << "Total size " << totalCount;
}

TEST(SR, readMatrices) {
  DataConfig cfg;
  cfg.set_format(DataConfig::TEXT);
  cfg.set_text(DataConfig::LIBSVM);
  cfg.add_file(rcvDataPath);
  auto res = searchFiles(cfg);
  StreamReader<float> reader(res);
  vector<Example> examples;
  int size = 20000;
  MatrixPtrList<float> matrices;
  //  while (reader.readMatrices(size, &matrices)) {
  //    LOG(INFO) << "size " << matrices.size();
  //  }
}
