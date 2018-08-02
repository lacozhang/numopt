/*
 * =====================================================================================
 *
 *       Filename:  bitmap.h
 *
 *    Description:  bitmap impl
 *
 *        Version:  1.0
 *        Created:  08/01/2018 14:38:06
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
class Bitmap;
typedef std::shared_ptr<Bitmap> BitmapPtr;

#define BITCOUNT_(x) (((BX_(x) + (BX_(x) >> 4)) & 0x0F0F0F0F) % 255)
#define BX_(x)                                                                 \
  ((x) - (((x) >> 1) & 0x77777777) - (((x) >> 2) & 0x33333333) -               \
   (((x) >> 3) & 0x11111111))

class Bitmap {
public:
  Bitmap() {}
  Bitmap(uint32 size, bool value = false) { resize(size, value); }
  
  void resize(uint32 size, bool value = false) {
    CHECK_GT(size, 0) << "size of bitmap must larger than 0";
    size_ = size;
    mapSize_ = (size_ >> kBitmapShift) + 1;
    map_ = new uint16[mapSize_];
    fill(value);
  }
  
  void fill(bool value) {
    memset(map_, value ? 0xFF : 0x00, mapSize_*sizeof(uint16));
  }
  
  void clear() {
    delete [] map_;
    map_ = nullptr;
    mapSize_ = 0;
    size_ = 0;
  }
  
  void set(uint32 i) {
    map_[i >> kBitmapShift] |= static_cast<uint16>(1 << (i & kBitmapMask));
  }
  
  void clear(uint32 i) {
    map_[i >> kBitmapShift] &= ~ static_cast<uint16>(1 << (i & kBitmapMask));
  }
  
  bool test(uint32 i) const {
    return static_cast<bool>((map_[i >> kBitmapShift] >> (i & kBitmapMask)) & 0x1);
  }
  
  bool operator[](uint32 i) const {
    return test(i);
  }
  
  uint32 size() const {
    return size_;
  }
  
  size_t memSize() const {
    return mapSize_ * sizeof(uint16);
  }

  ~Bitmap() { clear(); }
  
  uint32 nnz() {
    if(!initnnz_) {
      initnnz_ = true;
      for(int i=0; i<65536; ++i) {
        lookupTable_[i] = (uint8) BITCOUNT_(i);
      }
    }
    
    uint32 ret = 0;
    uint32 end = size_ >> kBitmapShift;
    for(int i=0; i < end; ++i) {
      ret += lookupTable_[map_[i]];
    }
    return ret + nnz(end << kBitmapShift, size_);
  }

private:
  
  uint32 nnz(uint32 start, uint32 end) {
    int ret = 0;
    for(uint32 i=start; i<end; ++i) {
      ret += (*this)[i];
    }
    return ret;
  }
  
  uint16 *map_ = nullptr;
  // size of real uint16 used.
  uint32 mapSize_ = 0;
  // size of bitmap
  uint32 size_ = 0;

  static const uint32 kBitmapShift = 4;
  static const uint32 kBitmapMask = 0x0F;

  uint8 lookupTable_[65536] = {0};
  bool initnnz_ = false;
};
} // namespace mltools
