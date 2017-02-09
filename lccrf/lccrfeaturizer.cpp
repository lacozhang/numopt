#include <iostream>
#include <fstream>
#include <string>
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "lccrfeaturizer.h"

const std::regex LccrFeaturizer::kWSRegex("\\s+", std::regex::icase);

bool LccrFeaturizer::AccumulateFeatures(const std::string& featsrc){

	namespace fs = boost::filesystem;
	if (!crftemplates_.IsValid()){
		BOOST_LOG_TRIVIAL(fatal) << "Crf templates load failed";
		return false;
	}

	fs::path filepath(featsrc);
	if (!fs::exists(filepath) || !fs::is_regular_file(filepath)){
		BOOST_LOG_TRIVIAL(fatal) << "Source file " << featsrc
			<< " do no exists or is not a regular file";
		return false;
	}

	std::ifstream src(featsrc);
	if (!src.is_open()){
		BOOST_LOG_TRIVIAL(fatal) << "Open file " << featsrc
			<< " failed";
		return false;
	}
	std::vector<std::string> sentence;
	std::string line;
	std::getline(src, line);
	while (src.good()){
		boost::algorithm::trim(line);
		if (line.size() < 1){

		}

		std::getline(src, line);
	}
}


bool LccrFeaturizer::FromLineToRawFeatures(
	const std::vector<std::string>& lines, 
	std::vector<std::vector<std::string>>& rawfeats){

}