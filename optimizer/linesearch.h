#pragma once

#ifndef __LINE_SEARCH_H__
#define __LINE_SEARCH_H__
#include <string>

class LineSearcher {
public:
	LineSearcher(const std::string& lsfuncstr, const std::string& lscondstr, int maxtries);
	~LineSearcher();

private:
	int maxtries_;
	int itercnt_;
};

#endif // !__LINE_SEARCH_H__
