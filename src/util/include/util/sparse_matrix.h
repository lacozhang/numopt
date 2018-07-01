#pragma once

#ifndef __SPARSE_MATRIX_H__
#define __SPARSE_MATRIX_H__

#include <thread>
#if GOOGLE_HASH
#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>
#endif
#include "range.h"
#include "util/common.h"
#include "util/dynamic_array.h"
#include "util/matrix.h"

namespace mltools {
template <typename I, typename V> class SparseMatrix;
template <typename I, typename V>
using SparseMatrixPtr = std::shared_ptr<SparseMatrix<I, V>>;

template <typename I, typename V> class SparseMatrix : public Matrix<V> {
public:
  USING_MATRIX;
  SparseMatrix() {}
  SparseMatrix(const MatrixInfo &info,const DArray<size_t> &offset,const DArray<I> &index,
               DArray<V> value): Matrix<V>(info, value), offset_(offset), index_(index){
    
  }
private:
  DArray<size_t> offset_;
  DArray<I> index_;
};

} // namespace mltools

#endif // __SPARSE_MATRIX_H__
