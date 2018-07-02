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
  SparseMatrix(const MatrixInfo &info, const DArray<size_t> &offset,
               const DArray<I> &index, DArray<V> value)
      : Matrix<V>(info, value), offset_(offset), index_(index) {}
  
  virtual void times(V *x, V *y) const override {
    templateTimes(x, y);
  }
  
  virtual MatrixPtr<V> dotTimes(const Matrix<V> &B) const override;
  
  virtual MatrixPtr<V> trans() const override {
    auto sm = std::make_shared<SparseMatrix<I,V>>(*this);
    sm->tranposeInfo();
    return sm;
  }
  
  virtual MatrixPtr<V> alterStorage() const override;
  virtual std::string debugString() const override;
  
  virtual bool writeToBinFile(std::string name) const override {
    return (writeProtoToASCIIFile(info_, name+".info") &&
            offset_.WriteToFile(name+".offset") &&
            index_.WriteToFile(name+".index") &&
            (binary() || value_.WriteToFile(name+".value")));
  }
  
  virtual MatrixPtr<V> colBlock(SizeR range) const override;
  virtual MatrixPtr<V> rowBlock(SizeR range) const override;
  
  bool binary() const {
    return info_.type() == MatrixInfo::SPARSE_BINARY;
  }
  
  DArray<I> index() const { return index_; }
  DArray<size_t> offset() const { return offset_; }
  
  virtual size_t memSize() const {
    return offset_.memSize() + index_.memSize() + value_.size();
  }
  
  virtual void resize(size_t rows, size_t cols, size_t nnz, bool rowMajor) {
    assert(false);
  }

private:
  
  /**
   * @brief Matrix-Vector multiplication with range of rows only
   *
   * Matrix vector multiplication using only some ranges.
   */
  template <typename W>
  void rangeTimes(SizeR rowRange, const W *const x, W *y) const {
    if(rowMajor()) {
      for(auto rowIdx=rowRange.begin(); rowIdx < rowRange.end(); ++rowIdx) {
        y[rowIdx] = 0.0;
        if(offset_[rowIdx] == offset[rowIdx+1]) continue;
        if(binary()) {
          for(auto colOffset=offset_[rowIdx]; colOffset < offset_[rowIdx+1]; ++colOffset) {
            auto colIdx = index_[colOffset];
            y[rowIdx] += x[colIdx];
          }
        } else {
          for(auto colOffset=offset_[rowIdx]; colOffset < offset_[rowIdx+1]; ++colOffset) {
            auto colIdx = index_[colOffset];
            y[rowIdx] += value_[colOffset] * x[colIdx];
          }
        }
      }
    } else {
      std::memset(y + rowRange.begin(), 0, sizeof(W) * rowRange.size());
      for(auto colIdx=0; colIdx < offset_.size() -1; colIdx++) {
        if(offset_[colIdx] == offset_[colIdx+1]) continue;
        W xAtCol = x[colIdx];
        if(binary()) {
          for(auto rowOffset=offset_[colIdx]; rowOffset<offset_[colIdx+1]; ++rowOffset) {
            auto rowIdx = index_[rowOffset];
            if(rowRange.contains(rowIdx)) {
              y[rowIdx] += xAtCol;
            }
          }
        } else {
          for(auto rowOffset=offset_[colIdx]; rowOffset<offset_[colIdx+1]; ++rowOffset) {
            auto rowIdx = index_[rowOffset];
            if(rowRange.contains(rowIdx)) {
              y[rowIdx] += xAtCol * value_[rowOffset];
            }
          }
        }
      }
    }
  }
  
  
  
  DArray<size_t> offset_;
  DArray<I> index_;
};

} // namespace mltools

#endif // __SPARSE_MATRIX_H__
