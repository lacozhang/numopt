/*
 * =====================================================================================
 *
 *       Filename:  parallel_sort.h
 *
 *    Description:  parallel sort
 *
 *        Version:  1.0
 *        Created:  07/27/2018 17:57:52
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "util/dynamic_array.h"

namespace mltools {

namespace {
template <typename T, class Fn>
void ParallelSort(T *data, size_t len, size_t grainsize, const Fn &cmp) {
  if (len <= grainsize) {
    std::sort(data, data + len, cmp);
  } else {
    std::thread thr(ParallelSort<T, Fn>(data, len / 2, grainsize, cmp));
    ParallelSort(data + len / 2, len - len / 2, grainsize, cmp);
    thr.join();
    std::inplace_merge(data, data + len / 2, data + len, cmp);
  }
}
} // namespace

template <typename T, class Fn>
void ParallelSort(DArray<T> *arr, int numThreads, const Fn &cmp) {
  CHECK_GT(numThreads, 0);
  size_t grainsize = std::max(arr->size() / numThreads + 5, (size_t)16 * 1024);
  ParallelSort(arr->data(), arr->size(), grainsize, cmp);
}
} // namespace mltools
