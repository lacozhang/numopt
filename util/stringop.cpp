#include "stringop.h"


void Split(const std::string& s, std::vector<std::string>& segs, const char* delim, bool skip) {
	Split(s.c_str(), s.size(), segs, delim, skip);
}

void Split(const char* buffer, const size_t len, std::vector<std::string>& segs, const char* delim, bool skip) {
	bool hits[256] = { false };
	size_t delimlen = std::strlen(delim), wordlen = 0;
	for (int i = 0; i < 256; ++i) hits[i] = false;
	for (int i = 0; i < delimlen; ++i) hits[delim[i]] = true;

	const char* start = buffer;
	for (int i = 0; i < len; ++i) {
		if (hits[buffer[i]] || (i == len - 1)) {
			const char* end = buffer + i;
			if (!hits[buffer[i]]) end++;
			if ((end == start) && (!skip)) {
				segs.push_back("");
			}
			else if (end > start) {
				std::string tmp(start, end);
				segs.push_back(tmp);
			}
			start = end + 1;
		}
	}

	if (hits[buffer[len - 1]] && !skip) {
		segs.push_back("");
	}
}