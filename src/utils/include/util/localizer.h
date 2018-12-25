/*
 * =====================================================================================
 *
 *       Filename:  localizer.h
 *
 *    Description:  localizer
 *
 *        Version:  1.0
 *        Created:  07/27/2018 17:58:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "util/crc32c.h"
#include "util/dynamic_array_impl.h"
#include "util/integral_types.h"
#include "util/parallel_sort.h"
#include "util/sparse_matrix.h"
#include <limits>

namespace mltools {

/// @brief remap the sparse matrix by chaing it's column number
template <typename I, typename V> class Localizer {
public:
  Localizer() {}
  ~Localizer() {}

  /**
   * @brief find the unique index & its count
   *
   *  Generally the count of the index will be used to filter some of the index
   * according to count.
   */
  template <typename C>
  void countUniqIndex(const DArray<I> &idx, DArray<I> *uniqIdx,
                      DArray<C> *idxFreq);

  template <typename C>
  void countUniqIndex(const MatrixPtr<V> &mat, DArray<I> *uniqIdx,
                      DArray<C> *idxFreq) {
    mat_ = std::static_pointer_cast<SparseMatrix<I, V>>(mat);
    countUniqIndex(mat_->index(), uniqIdx, idxFreq);
  }

  /**
   * @brief re-map the input sparse matrix to relative dense matrix.
   */
  MatrixPtr<V> remapIndex(const DArray<I> &idxDict);

  void clear() { pair_.clear(); }

  size_t memSize() {
    return pair_.size() * sizeof(Pair) +
           (mat_ == nullptr ? 0 : mat_->memSize());
  }

  /**
   * @brief re-construct the matrix according to idxdict
   *
   */
  MatrixPtr<V> remapIndex(const MatrixInfo &info, const DArray<size_t> &offset,
                          const DArray<I> &index, const DArray<V> &value,
                          const DArray<I> &idxdict) const;

private:
#pragma pack(push)
#pragma pack(4)
  struct Pair {
    I k_;
    uint32 i_;
  };
#pragma pack(pop)
  DArray<Pair> pair_;
  SparseMatrixPtr<I, V> mat_;
};

template <typename I, typename V>
template <typename C>
void Localizer<I, V>::countUniqIndex(const DArray<I> &idx, DArray<I> *uniqIdx,
                                     DArray<C> *idxFreq) {
  if (idx.empty()) {
    return;
  }
  CHECK_NOTNULL(uniqIdx);
  CHECK_LT(idx.size(), kuint32max) << "size too large ";
  CHECK_GT(FLAGS_num_threads, 0);

  pair_.resize(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    pair_[i].k_ = idx[i];
    pair_[i].i_ = i;
  }

  ParallelSort(&pair_, FLAGS_num_threads,
               [](const Pair &a, const Pair &b) { return a.k_ < b.k_; });
  uniqIdx->clear();
  if (idxFreq != nullptr) {
    idxFreq->clear();
  }

  uint32 cntMax = static_cast<uint32>(std::numeric_limits<C>::max());
  I curr = pair_[0].k_;
  uint32 cnt = 0;
  for (const Pair &v : pair_) {
    if (v.k_ != curr) {
      uniqIdx->push_back(curr);
      curr = v.k_;
      if (idxFreq) {
        C tcnt = static_cast<C>(std::min(cnt, cntMax));
        idxFreq->push_back(tcnt);
      }
      cnt = 0;
    }
    ++cnt;
  }

  uniqIdx->push_back(curr);
  if (idxFreq) {
    C tcnt = static_cast<C>(std::min(cnt, cntMax));
    idxFreq->push_back(tcnt);
  }
}

template <typename I, typename V>
MatrixPtr<V> Localizer<I, V>::remapIndex(const DArray<I> &idxdict) {
  CHECK(mat_);
  return remapIndex(mat_->info(), mat_->offset(), mat_->index(), mat_->value(),
                    idxdict);
}

template <typename I, typename V>
MatrixPtr<V> Localizer<I, V>::remapIndex(const MatrixInfo &info,
                                         const DArray<size_t> &offset,
                                         const DArray<I> &index,
                                         const DArray<V> &value,
                                         const DArray<I> &idxdict) const {
  if (index.empty() || idxdict.empty()) {
    return MatrixPtr<V>();
  }
  CHECK_NE(info.type(), MatrixInfo::DENSE) << "dense matrix is compact already";
  CHECK_LT(idxdict.size(), kuint32max);
  CHECK_EQ(offset.back(), index.size());
  CHECK_EQ(index.size(), pair_.size());
  bool bin = value.empty();
  if (!bin) {
    CHECK_EQ(value.size(), index.size());
  }

  uint32 matched = 0;
  DArray<uint32> remappedIndex(pair_.size(), 0);
  const I *currDict = idxdict.begin();
  const Pair *currPair = pair_.begin();
  while (currDict != idxdict.end() && currPair != pair_.end()) {
    if (*currDict < currPair->k_) {
      ++currDict;
    } else {
      if (*currDict == currPair->k_) {
        remappedIndex[currPair->i_] = (uint32)(currDict - idxdict.begin()) + 1;
        ++matched;
      }
      ++currPair;
    }
  }

  DArray<I> newIndex(matched);
  DArray<size_t> newOffset(offset.size());
  newOffset[0] = 0;
  DArray<V> newValue(std::min(value.size(), (size_t)matched));

  size_t k = 0;
  for (size_t i = 0; i < (offset.size() - 1); ++i) {
    size_t n = 0;
    for (size_t j = offset[i]; j < offset[i + 1]; ++j) {
      if (remappedIndex[j] == 0) {
        continue;
      }
      ++n;
      if (!bin) {
        newValue[k] = value[j];
      }
      newIndex[k++] = remappedIndex[j] - 1;
    }
    newOffset[i + 1] = newOffset[i] + n;
  }

  CHECK_EQ(k, matched);
  auto newInfo = info;
  newInfo.set_sizeof_idx(sizeof(uint32));
  newInfo.set_nnz(newIndex.size());
  SizeR local(0, idxdict.size());
  if (newInfo.row_major()) {
    local.to(newInfo.mutable_col());
  } else {
    local.to(newInfo.mutable_row());
  }

  return MatrixPtr<V>(
      new SparseMatrix<I, V>(newInfo, newOffset, newIndex, newValue));
}
} // namespace mltools
