#pragma once

#include <string>
#include <vector>

#ifndef __STRINGOP_H__
#define __STRINGOP_H__
namespace Util {

void Split(const std::string &s, std::vector<std::string> &segs,
           const char *delim, bool skip);

void Split(const unsigned char *buffer, size_t len,
           std::vector<std::string> &segs, const unsigned char *delim,
           bool skip);
} // namespace Util

#endif // __STRINGOP_H__
