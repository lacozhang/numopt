#include <iostream>
#include <vector>
#include <string>

#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>

int main(int argc, const char* argv[]) {

	std::vector<std::string> featsrcs, inputfiles;
	std::string outputprefix, featemplate;
	namespace po = boost::program_options;
	po::options_description desc("Options for CRF featurization");
	desc.add_options()
		("help,h", "print out help message")
		("feat,f", po::value<std::vector<std::string>>(&featsrcs)->required(), "source file used to extract feature")
		("files,i", po::value<std::vector<std::string>>(&inputfiles), "files need featurization")
		("output,o", po::value<std::string>(&outputprefix)->required(), "required outout prefix")
		("template,t", po::value<std::string>(&featemplate), "templates for crf feature");
	po::variables_map vm;
	try
	{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch (const po::required_option& e){
		std::cerr << e.what() << std::endl;
		std::cout << desc;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cout << desc;
	}
	

	return 0;
}