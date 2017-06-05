#ifndef __STRINGOP_H__
#define __STRINGOP_H__
#include <string>
#include <vector>
#include <sstream>

void Split(const std::string& s, std::vector<std::string>& segs, const char* delim){
	Split(s.c_str(), segs, delim, skip);
}

void Split(const char* buffer, std::vector<std::string>& segs, const char* delim){
	bool hits[256] = { 0 };
	int delimlen = std::strlen(delim), wordlen = 0;
	for (int i = 0; i < delimlen; ++i){
		hits[delim[i]] = true;
	}

	const char *start = buffer, *end = nullptr;
	while (*start != '\0'){
		while (*start != '0' && hits[*start]) ++start;
		end = start;
		while (*end != '\0' && !hits[*end]) ++end;
		wordlen = end - start;
		if (wordlen < 1) break;
		std::string word(wordlen, '\0');
		for (int i = 0; i < wordlen; ++i)
			word[i] = start[i];
		segs.push_back(word);
		start = end;
	}
}


#endif // __STRINGOP_H__