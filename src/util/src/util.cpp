#include "util/util.h"

timeutil::timeutil(){
}

void timeutil::tic(){
	counter_ = std::chrono::system_clock::now();
}

double timeutil::toc(){
	std::chrono::duration<double> t = std::chrono::system_clock::now() - counter_;
	return t.count();
}


boost::program_options::variables_map ParseArgs(int argc, const char* argv[],
	boost::program_options::options_description& optionsdesc, bool allowunreg) {
	boost::program_options::variables_map vm;
	try {
		if (allowunreg) {
			boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(optionsdesc).allow_unregistered().run(), vm);
		}
		else {
			boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(optionsdesc).run(), vm);
		}
		boost::program_options::notify(vm);
	}
	catch (std::exception& e) {
		BOOST_LOG_TRIVIAL(error) << "parse command line failed :" << e.what();
	}

	return vm;
}

int aligned_by(const int val, const int alignment) {
    return  ((val + alignment - 1) / alignment)*alignment;
}
