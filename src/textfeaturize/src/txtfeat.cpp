#include "dataop/dataop.h"
#include "util/util.h"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <climits>
#include <fstream>
#include <iostream>

namespace triv = boost::log::trivial;

int main(int argc, char *argv[]) {

  namespace po = boost::program_options;
  BOOST_LOG_TRIVIAL(trace) << "Start Running program";

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
    BOOST_LOG_TRIVIAL(trace) << "Input File  :" << infile;
    BOOST_LOG_TRIVIAL(trace) << "Output File :" << outfile;

    boost::shared_ptr<DataSamples> samples;
    boost::shared_ptr<LabelVector> labels;

    BOOST_LOG_TRIVIAL(trace) << "Load text format data";
    load_libsvm_data(infile, samples, labels, true, 0);
    BOOST_LOG_TRIVIAL(trace) << "Save to Bin format";
    save_libsvm_data_bin(outfile, samples, labels);
    BOOST_LOG_TRIVIAL(trace) << "Verify the bin format by reloading";
    load_libsvm_data(outfile, samples, labels, true, 0);
  } catch (std::exception &e) {
    BOOST_LOG_TRIVIAL(fatal) << e.what();
  }

  return 0;
}