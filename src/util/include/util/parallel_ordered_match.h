/*
 * =====================================================================================
 *
 *       Filename:  parallel_ordered_match.h
 *
 *    Description:  sorting function
 *
 *        Version:  1.0
 *        Created:  07/24/2018 19:43:08
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "util/assign_op.h"
#include "util/dynamic_array.h"

namespace mltools {

template <typename K, typename V>
void ParallelOrderedMatch(const K *srcKey, const K *srcKeyEnd, const V *srcVal,
                          const K *dstKey, const K *dstKeyEnd, V *dstVal, int k,
                          AssignOpType op, size_t grainsize, size_t *n) {
  size_t srcLen = std::distance(srcKey, srcKeyEnd);
  size_t dstLen = std::distance(dstKey, dstKeyEnd);
  if (srcLen == 0 || dstLen == 0) {
    return;
  }
  srcKey = std::lower_bound(srcKey, srcKeyEnd, *dstKey);
  srcVal += (srcKey - (srcKeyEnd - srcLen)) * k;
  if (dstLen <= grainsize) {
    while (dstKey != dstKeyEnd && srcKey != srcKeyEnd) {
      if (*srcKey < *dstKey) {
        ++srcKey;
        srcVal += k;
      } else {
        if (!(*srcKey < *dstKey)) {
          for (int i = 0; i < k; ++k) {
            AssignOp(dstVal[i], srcVal[i], op);
          }
          ++srcKey;
          srcVal += k;
          *n += k;
        }
        ++dstKey;
        dstVal += k;
      }
    }
  } else {
    std::thread thr(ParallelOrderedMatch<K, V>, srcKey, srcKeyEnd, srcVal,
                    dstKey, dstKey + dstLen / 2, dstVal, k, op, grainsize, n);
    size_t m = 0;
    ParallelOrderedMatch<K, V>(srcKey, srcKeyEnd, srcVal, dstKey + dstLen / 2,
                               dstKeyEnd, dstVal + (dstLen / 2) * k, k, op,
                               grainsize, &m);
    thr.join();
    *n += m;
  }
}

template <typename K, typename V>
size_t ParallelOrderedMatch(
    const DArray<K> &srcKey, // source keys
    const DArray<V> &srcVal, // source values
    const DArray<K> &dstKey, // destination keys
    DArray<V> *dstVal,       // destination values
    int k = 1,               // the size of value entry = k*sizeof(V)
    AssignOpType op = AssignOpType::ASSIGN, // assignment operator
    int numThreads = FLAGS_num_threads) {
  CHECK_GT(numThreads, 0);
  CHECK_EQ(srcKey.size() * k, srcVal.size());
  if (dstVal->empty()) {
    dstVal->resize(dstKey.size() * k);
    dstVal->setZero();
  } else {
    CHECK_EQ(dstVal->size(), dstKey.size() * k);
  }
  SizeR range = dstKey.findRange(srcKey.range());
  size_t grainsize =
      std::max(range.size() * k / numThreads + 5, (size_t)1024 * 1024);
  size_t n = 0;
  ParallelOrderedMatch<K, V>(
      srcKey.begin(), srcKey.end(), srcVal.begin(),
      dstKey.begin() + range.begin(), dstKey.begin() + range.end(),
      dstVal->begin() + range.begin() * k, k, op, grainsize, &n);
  return n;
}

template <typename K, typename V>
void ParallelUnion(const DArray<K> &key1, const DArray<V> &val1,
                   const DArray<K> &key2, const DArray<V> &val2,
                   DArray<K> *joinedKey, DArray<V> *joinedVal, int k = 1,
                   AssignOpType op = AssignOpType::PLUS,
                   int numThreads = FLAGS_num_threads) {
  *(CHECK_NOTNULL(joinedKey)) = key1.setUnion(key2);
  CHECK_NOTNULL(joinedVal)->resize(0);

  auto n1 = ParallelOrderedMatch<K, V>(key1, val1, *joinedKey, joinedVal, k, op,
                                       numThreads);
  CHECK_EQ(n1, key1.size());

  auto n2 = ParallelOrderedMatch<K, V>(key2, val2, *joinedKey, joinedVal, k, op,
                                       numThreads);
  CHECK_EQ(n2, key2.size());
}
} // namespace mltools
