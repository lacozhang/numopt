#include <iostream>
#include <vector>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>

#include "lccrfeaturizer.h"

int main(int argc, const char* argv[]) {

	namespace fs = boost::filesystem;
	std::vector<std::string> featsrcs, inputfiles;
	std::string outputprefix, featemplate;
	int cutoff;
	namespace po = boost::program_options;
	po::options_description desc("Options for CRF featurization");
	desc.add_options()
		("help,h", "print out help message")
		("feat,f", po::value<std::vector<std::string>>(&featsrcs)->required(), "source file used to extract feature")
		("files,i", po::value<std::vector<std::string>>(&inputfiles), "files need featurization")
		("output,o", po::value<std::string>(&outputprefix)->required(), "required outout prefix")
		("template,t", po::value<std::string>(&featemplate)->required(), "templates for crf feature")
		("cutoff,c", po::value<int>(&cutoff)->default_value(1), "cutoff values for feature extraction");
	po::variables_map vm;
	try
	{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch (const po::required_option& e){
		std::cerr << e.what() << std::endl;
		std::cout << desc;
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cout << desc;
		return 0;
	}

	if (vm.count("h") || vm.count("help")) {
		std::cout << desc;
		return 0;
	}

	LccrFeaturizer featurizer(featemplate);
	for (const std::string& src : featsrcs) {
		featurizer.AccumulateFeatures(src, cutoff, cutoff);
	}

	for (std::string& src : inputfiles) {
		std::string binfilepath = src + ".bin";
		featurizer.FeaturizeFile(src, binfilepath);
	}

	featurizer.Save(outputprefix);

	return 0;
}