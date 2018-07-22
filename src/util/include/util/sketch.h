/*
 * =====================================================================================
 *
 *       Filename:  sketch.h
 *
 *    Description:  sketch function
 *
 *        Version:  1.0
 *        Created:  07/22/2018 17:33:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "util/common.h"

namespace mltools {
/// @brief the base classes for bloom filter/counter min
class Sketch {
public:
protected:
  uint32 hash(const uint64 &key) const {
    const uint32 seed = 0xbc9f1d34;
    const uint32 m = 0xc6a4a793;
    const uint32 n = 8; // sizeof uint64
    uint32 h = seed ^ (n * m);

    uint32 w = (uint32)key;
    h += w;
    h *= m;
    h ^= (h >> 16);

    w = (uint32)(key >> 32);
    h += w;
    h *= m;
    h ^= (h >> 16);
    return h;
  }
};
} // namespace mltools
