/*
 * =====================================================================================
 *
 *       Filename:  frequency_filter.h
 *
 *    Description:  filter data by frequency
 *
 *        Version:  1.0
 *        Created:  07/22/2018 15:53:42
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once
#include "util/countmin.h"
#include "util/dynamic_array.h"

namespace mltools {
  
  /// @brief filter in-frequent keys through countmin sketch.
  template <typename K, typename V>
  class FrequencyFilter {
  public:
    void insertKeys(const DArray<K> &keys, const DArray<V> &counts);
    
    DArray<K> queryKeys(const DArray<K> &keys, int threshold);
    
    bool empty() { return count_.empty(); }
    
    void resize(int n, int k) {
      count_.resize(n, k, 254);
    }
    
    void clear() {
      count_.clear();
    }
  private:
    CountMin<K, V> count_;
  };
  
  
  template <typename K, typename V>
  DArray<K> FrequencyFilter<K,V>::queryKeys(const DArray<K> &key, int threshold) {
    CHECK_LT(threshold, kuint8max) << "change to uint16 or uint32 ...";
    DArray<K> filteredKey;
    for(auto &k: key) {
      if((int)count_.query(k) > threshold) {
        filteredKey.push_back(k);
      }
    }
    return filteredKey;
  }
  
  template <typename K, typename V>
  void FrequencyFilter<K,V>::insertKeys(const DArray<K> &keys, const DArray<V> &counts) {
    CHECK_EQ(keys.size(), counts.size());
    for(size_t i=0; i<keys.size(); ++i) {
      count_.insert(keys[i], counts[i]);
    }
  }
}
