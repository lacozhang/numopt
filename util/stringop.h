#pragma once

#ifndef __STRINGOP_H__
#define __STRINGOP_H__
#include <string>
#include <vector>
#include <sstream>

void Split(const std::string& s, std::vector<std::string>& segs, const char* delim, bool skip);

void Split(const char* buffer, size_t len, std::vector<std::string>& segs, const char* delim, bool skip);


#endif // __STRINGOP_H__