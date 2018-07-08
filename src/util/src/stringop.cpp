#include "util/stringop.h"
#include <cstring>

namespace Util {
void Split(const std::string &s, std::vector<std::string> &segs,
           const char *delim, bool skip) {
  Split((const unsigned char *)s.c_str(), s.size(), segs,
        (const unsigned char *)delim, skip);
}

void Split(const unsigned char *buffer, const size_t len,
           std::vector<std::string> &segs, const unsigned char *delim,
           bool skip) {
  if (len == 0)
    return;
  bool hits[256] = {false};
  size_t delimlen = std::strlen((const char *)delim);
  for (int i = 0; i < 256; ++i)
    hits[i] = false;
  for (int i = 0; i < delimlen; ++i)
    hits[delim[i]] = true;

  const unsigned char *start = buffer;
  for (int i = 0; i < len; ++i) {
    if (hits[buffer[i]] || (i == len - 1)) {
      const unsigned char *end = buffer + i;
      if (!hits[buffer[i]])
        end++;
      if ((end == start) && (!skip)) {
        segs.push_back("");
      } else if (end > start) {
        std::string tmp(start, end);
        segs.push_back(std::move(tmp));
      }
      start = end + 1;
    }
  }

  if (hits[buffer[len - 1]] && !skip) {
    segs.push_back("");
  }
}

std::string join(const std::vector<std::string> &elems,
                 const std::string delim) {
  std::string str;
  for (auto elem : elems) {
    if (!str.empty()) {
      str += delim;
    }
    str += elem;
  }
  return str;
}
} // namespace Util
