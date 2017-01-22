#pragma once

#ifndef __CRFTEMPLATE_H__
#define __CRFTEMPLATE_H__
#include <vector>
#include <string>
#include <tuple>
#include <regex>

class CrfTemplate {
public:
	CrfTemplate(std::string filepath);
	~CrfTemplate() {}
	void ExtractUnigramFeatures(const std::vector<std::vector<std::string>>& raw, std::vector<std::vector<std::string>>& features);
	void ExtractBigramFeatures(const std::vector<std::string>& raw, std::vector<std::vector<std::string>>& features);

private:

	bool LoadTemplate(std::string);
	bool IsLccrfTemplate(std::string& line);
	bool IsTriggerConstantFeature(const std::tuple<int, int>& featsepc);

	void ParseLCCRFTemplateLine(const std::string& featspecs, std::vector<std::tuple<int, int>>& ret);
	void HandleLCCRFTemplateLine(std::string& line, std::vector<std::tuple<int, int>>& feats);

	bool valid_;
	std::vector<std::vector<std::tuple<int, int>>> unigrams_;
	std::vector<std::vector<std::tuple<int, int>>> bigrams_;

	static const std::regex kLccrfRegex;
};

#endif // !__CRFTEMPLATE_H__
