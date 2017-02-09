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

bool CrfTemplate::ExtractUnigramFeatures(
	const std::vector<std::vector<std::string>>& raw, 
	std::vector<std::vector<std::string>>& features)
{
	features.clear();
	if (valid_) {
		for (int i = 0; i < raw.size(); ++i) {

			std::vector<std::string> currfeats;
			for (const LccrfTemplateLine& templatelines : unigrams_) {

				std::vector<std::string> currfeat;
				currfeat.clear();
				currfeat.push_back(templatelines.RetrieveFeaturePrefix());
				ExtractAsTemplateLine(raw, currfeat, templatelines, i);
				if (currfeat.size() > 1) {
					currfeats.push_back((boost::algorithm::join(currfeat, "_")));
				}
			}
			features.push_back(currfeats);
		}
	}
	else {
		BOOST_LOG_TRIVIAL(fatal) << "Load tempalte failed!!!";
		return false;
	}
	return true;
}

bool CrfTemplate::ExtractBigramFeatures(
	const std::vector<std::vector<std::string>>& raw, 
	std::vector<std::vector<std::string>>& features){
	if (valid_){
		features.clear();
		
		for (int currpos = 0; currpos < raw.size(); ++currpos){
			std::vector<std::string> currfeats;
			currfeats.clear();
			for (const LccrfTemplateLine& templine : bigrams_){
				std::vector<std::string> currfeat;
				currfeat.clear();
				currfeat.push_back(templine.RetrieveFeaturePrefix());
				ExtractAsTemplateLine(raw, currfeat, templine, currpos);

				if (currfeat.size() > 1){
					currfeats.push_back(boost::algorithm::join(currfeat, "_"));
				}
			}
			features.push_back(currfeats);
		}
	}
	else {
		BOOST_LOG_TRIVIAL(fatal) << "Load template failed!!!";
		return false;
	}
	return true;
}

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
		while (src.good())
		{

			boost::algorithm::erase_all(line, " \n\r\t");
			if (!boost::algorithm::starts_with(line, "#")) {
				LccrfTemplateLine featspec;
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

void CrfTemplate::HandleLCCRFTemplateLine(std::string & line, LccrfTemplateLine& featspec)
{
	ParseLCCRFTemplateLine(line, featspec);
	if (featspec.RetrieveTemplate().size() > 0) {
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

void CrfTemplate::ExtractAsTemplateLine(const std::vector<std::vector<std::string>>& rawfeatures, std::vector<std::string>& extracted, const LccrfTemplateLine & line, int currentpos)
{
	for (const std::tuple<int, int>& item : line.RetrieveTemplate()) {
		int rowidx = currentpos + std::get<0>(item), colidx = std::get<1>(item);
		if (rowidx < rawfeatures.size() && rowidx >= 0) {
			if (colidx >= 0 && colidx < rawfeatures[rowidx].size()) {
				extracted.push_back(rawfeatures[rowidx][colidx]);
			}
		}
	}
}

void CrfTemplate::ParseLCCRFTemplateLine(const std::string& line, LccrfTemplateLine& featspecs)
{
	if (line[0] != 'U' && line[0] != 'B') {
		BOOST_LOG_TRIVIAL(fatal) << "LCCRF tempalte line format error : " << line;
		std::exit(1);
	}

	std::vector<std::string> comps;
	std::vector<std::string> featspecstrs;
	featspecs.Clear();

	boost::algorithm::split(comps, line, boost::algorithm::is_any_of(":"));

	featspecs.SetFeaturePrefix(comps[0]);
	if (comps.size() == 1) {
		featspecs.AddTemplatePart(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
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
			try {
				rowidx = boost::lexical_cast<int>(m[1].str());
				colidx = boost::lexical_cast<int>(m[2].str());
				safe = true;
			}
			catch (const boost::bad_lexical_cast& e) {
				BOOST_LOG_TRIVIAL(warning) << e.what() << " format error: " << featstr;
			}

			if (safe) {
				featspecs.AddTemplatePart(rowidx, colidx);
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
