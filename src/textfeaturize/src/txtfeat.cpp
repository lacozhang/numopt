#include "dataop/dataop.h"
#include "util/util.h"
#include <boost/program_options.hpp>
#include <climits>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {

  namespace po = boost::program_options;
  LOG(INFO) << "Start Running program";

  try {
    po::options_description desc("make program cache");
    desc.add_options()("help,h", "print help message")(
        "in,i", po::value<std::string>()->required(), "input file name")(
        "out,o", po::value<std::string>()->required(), "output file name");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    std::string infile, outfile;
    infile = vm["in"].as<std::string>();
    outfile = vm["out"].as<std::string>();
    LOG(INFO) << "Input File  :" << infile;
    LOG(INFO) << "Output File :" << outfile;

    boost::shared_ptr<DataSamples> samples;
    boost::shared_ptr<LabelVector> labels;

    LOG(INFO) << "Load text format data";
    load_libsvm_data(infile, samples, labels, true, 0);
    LOG(INFO) << "Save to Bin format";
    save_libsvm_data_bin(outfile, samples, labels);
    LOG(INFO) << "Verify the bin format by reloading";
    load_libsvm_data(outfile, samples, labels, true, 0);
  } catch (std::exception &e) {
    LOG(FATAL) << e.what();
  }

  return 0;
}
