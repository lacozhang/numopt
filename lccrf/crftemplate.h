#pragma once

#ifndef __CRFTEMPLATE_H__
#define __CRFTEMPLATE_H__
#include <vector>
#include <string>
#include <tuple>
#include <regex>
#include <utility>

class CrfTemplate {
public:
	CrfTemplate(std::string filepath);
	~CrfTemplate() {}
	bool ExtractUnigramFeatures(const std::vector<std::vector<std::string>>& raw, std::vector<std::vector<std::string>>& features);
	bool ExtractBigramFeatures(const std::vector<std::vector<std::string>>& raw, std::vector<std::vector<std::string>>& features);
	bool IsValid() const{
		return valid_;
	}
	bool LoadTemplate(std::string);
	void SaveToFile(const std::string& filepath);

	class LccrfTemplateLine {
	public:
		LccrfTemplateLine() {
			featprefix_.clear();
			featemplates_.clear();
		}

		~LccrfTemplateLine() {}

		void SetFeaturePrefix(const std::string& featname) {
			featprefix_ = featname;
		}

		const std::string& RetrieveFeaturePrefix() {
			return featprefix_;
		}

		const std::string& RetrieveFeaturePrefix() const {
			return featprefix_;
		}

		void AddTemplatePart(int row, int col) {
			featemplates_.push_back(std::tuple<int, int>(row, col));
		}
		
		const std::vector<std::tuple<int, int>>& RetrieveTemplate() const {
			return featemplates_;
		}

		const std::vector<std::tuple<int, int>>& RetrieveTemplate() {
			return featemplates_;
		}

		void Clear() {
			featprefix_.clear();
			featemplates_.clear();
		}

	private:
		std::string featprefix_;
		std::vector<std::tuple<int, int>> featemplates_;
	};

private:

	bool IsLccrfTemplate(std::string& line);
	bool IsTriggerConstantFeature(const std::tuple<int, int>& featsepc);

	void ParseLCCRFTemplateLine(const std::string& featspecs, LccrfTemplateLine& ret);
	void HandleLCCRFTemplateLine(std::string& line, LccrfTemplateLine& feats);

	void ExtractAsTemplateLine(const std::vector<std::vector<std::string>>& rawfeatures, std::vector<std::string>& extracted, const LccrfTemplateLine & line, int currentpos);
	bool SaveTemplateSet(std::ofstream& sink, const std::vector<LccrfTemplateLine>& featspecs);
	bool valid_;
	std::vector<LccrfTemplateLine> unigrams_;
	std::vector<LccrfTemplateLine> bigrams_;

	static const std::regex kLccrfRegex;
};

#endif // !__CRFTEMPLATE_H__
