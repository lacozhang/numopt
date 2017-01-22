#include <fstream>
#include <limits>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#include <boost/lexical_cast.hpp>
#include "crftemplate.h"


const std::regex CrfTemplate::kLccrfRegex("%x\\[(\\+?|-?[[:digit:]]+),(\\+?|-?[[:digit:]]+)\\]", std::regex::icase);

CrfTemplate::CrfTemplate(std::string filepath)
{
	valid_ = false;
	boost::filesystem::path f(filepath);
	if (!boost::filesystem::exists(f) || !boost::filesystem::is_regular_file(f)) {
		BOOST_LOG_TRIVIAL(fatal) << "template file " << filepath << " not exist or is not a regular file";
		return;
	}
	
	if (LoadTemplate(filepath)) {
		valid_ = true;
	}
}

void CrfTemplate::ExtractUnigramFeatures(const std::vector<std::vector<std::string>>& raw, std::vector<std::vector<std::string>>& features)
{
	if (valid_) {
		for (int i = 0; i < raw.size(); ++i) {

			std::vector<std::string> currfeats;
			for (const std::vector<std::tuple<int, int>>& templatelines : unigrams_) {

				std::vector<std::string> currfeat;
				currfeat.push_back("U");
				for (const std::tuple<int, int>& item : templatelines) {
					int rowidx = i + std::get<0>(item), colidx = std::get<1>(item);
					if (rowidx < raw.size() && rowidx >= 0) {
						if (colidx >= 0 && colidx < raw[rowidx].size()) {
							currfeat.push_back(raw[rowidx][colidx]);
						}
					}
				}

				if (currfeat.size() > 1) {
					currfeats.push_back((boost::algorithm::join(currfeat, "_")));
				}
			}
			features.push_back(currfeats);
		}
	}
	else {
		BOOST_LOG_TRIVIAL(fatal) << "Load tempalte failed!!!";
		std::exit(1);
	}
}

void 

bool CrfTemplate::LoadTemplate(std::string crftemplates)
{
	bool succ = false;
	std::ifstream src(crftemplates);
	std::string line;
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "Failed to open " << crftemplates;
	}
	else {
		std::getline(src, line);
		std::vector<std::tuple<int, int>> featspec;
		featspec.clear();
		while (src.good())
		{
			boost::algorithm::erase_all(line, " \n\r\t");
			if (!boost::algorithm::starts_with(line, "#")) {

				if (IsLccrfTemplate(line)) {
					HandleLCCRFTemplateLine(line, featspec);
				}
				else {
					BOOST_LOG_TRIVIAL(fatal) << "Not support other type of model tempalte yet";
				}
			}
			std::getline(src, line);
		}

		if (!src.eof()) {
			succ = false;
			BOOST_LOG_TRIVIAL(error) << "Unexpected end of templte file";
		}
		else {
			succ = true;
		}
	}

	return succ;
}

void CrfTemplate::HandleLCCRFTemplateLine(std::string & line, std::vector<std::tuple<int, int>>& featspec)
{
	ParseLCCRFTemplateLine(line, featspec);
	if (featspec.size() > 0) {
		switch (line[0])
		{
		case 'U':
			unigrams_.push_back(featspec);
			break;
		case 'B':
			bigrams_.push_back(featspec);
			break;
		default:
			break;
		}
	}
}

void CrfTemplate::ParseLCCRFTemplateLine(const std::string& line, std::vector<std::tuple<int, int>>& featspecs)
{
	if (line[0] != 'U' && line[0] != 'B') {
		BOOST_LOG_TRIVIAL(fatal) << "LCCRF tempalte line format error : " << line;
		std::exit(1);
	}

	std::vector<std::string> comps;
	std::vector<std::string> featspecstrs;
	featspecs.clear();

	boost::algorithm::split(comps, line, boost::algorithm::is_any_of(":"));

	if (comps.size() == 1) {
		featspecs.push_back(std::tuple<int, int>(std::numeric_limits<int>::max(), std::numeric_limits<int>::max()));
		return;
	}

	if (comps.size() != 2) {
		BOOST_LOG_TRIVIAL(fatal) << "LCCRF tempalte line format error : " << line;
		std::exit(1);
	}

	boost::algorithm::split(featspecstrs, comps[1], boost::algorithm::is_any_of("/"));
	if (featspecstrs.size() < 1) {
		BOOST_LOG_TRIVIAL(fatal) << "LCCRF tempalte line format error : " << line;
		std::exit(1);
	}

	for (std::string& featstr : featspecstrs) {
		std::smatch m;
		std::regex_match(featstr, m, kLccrfRegex);
		if (m.ready() && (!m.empty()) && (m.size() == 3)) {
			int rowidx = -1, colidx = -1;
			bool safe = false;
			try{
				rowidx = boost::lexical_cast<int>(m[1].str());
				colidx = boost::lexical_cast<int>(m[2].str());
				safe = true;
			}
			catch (const boost::bad_lexical_cast& e) {
				BOOST_LOG_TRIVIAL(warning) << e.what() << " format error: " << featstr;
			}

			if (safe) {
				featspecs.push_back(std::make_tuple(rowidx, colidx));
			}
		}
		else {
			BOOST_LOG_TRIVIAL(warning) << "LCCRFTemplate format error " << featstr;
		}
	}
}

bool CrfTemplate::IsTriggerConstantFeature(const std::tuple<int, int>& featsepc)
{
	return (std::get<0>(featsepc) == std::numeric_limits<int>::max()) &&
		(std::get<1>(featsepc) == std::numeric_limits<int>::max());
}

bool CrfTemplate::IsLccrfTemplate(std::string & line)
{
	return (line[0] == 'U') || (line[0] == 'B');
}
