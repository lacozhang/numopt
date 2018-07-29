#pragma once

#ifndef __SPARSE_MATRIX_H__
#define __SPARSE_MATRIX_H__

#include <gflags/gflags.h>
#include <thread>
#if GOOGLE_HASH
#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>
#endif
#include "range.h"
#include "util/common.h"
#include "util/dynamic_array.h"
#include "util/matrix.h"
#include "util/threadpool.h"

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

  virtual void times(const V *x, V *y) const override { templateTimes(x, y); }

  virtual MatrixPtr<V> dotTimes(const MatrixPtr<V> &B) const override;

  virtual MatrixPtr<V> trans() const override {
    auto sm = std::make_shared<SparseMatrix<I, V>>(*this);
    sm->tranposeInfo();
    return sm;
  }

  virtual MatrixPtr<V> alterStorage() override;
  virtual std::string debugString() const override;

  virtual bool writeToBinFile(std::string name) const override {
    return (writeProtoToASCIIFile(info_, name + ".info") &&
            offset_.writeToFile(name + ".offset") &&
            index_.writeToFile(name + ".index") &&
            (binary() || value_.writeToFile(name + ".value")));
  }

  virtual MatrixPtr<V> colBlock(SizeR range) const override;
  virtual MatrixPtr<V> rowBlock(SizeR range) const override;

  bool binary() const { return info_.type() == MatrixInfo::SPARSE_BINARY; }

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
    if (rowMajor()) {
      for (auto rowIdx = rowRange.begin(); rowIdx < rowRange.end(); ++rowIdx) {
        y[rowIdx] = 0.0;
        if (offset_[rowIdx] == offset_[rowIdx + 1])
          continue;
        if (binary()) {
          for (auto colOffset = offset_[rowIdx];
               colOffset < offset_[rowIdx + 1]; ++colOffset) {
            auto colIdx = index_[colOffset];
            y[rowIdx] += x[colIdx];
          }
        } else {
          for (auto colOffset = offset_[rowIdx];
               colOffset < offset_[rowIdx + 1]; ++colOffset) {
            auto colIdx = index_[colOffset];
            y[rowIdx] += value_[colOffset] * x[colIdx];
          }
        }
      }
    } else {
      std::memset(y + rowRange.begin(), 0, sizeof(W) * rowRange.size());
      for (auto colIdx = 0; colIdx < cols(); colIdx++) {
        if (offset_[colIdx] == offset_[colIdx + 1])
          continue;
        W xAtCol = x[colIdx];
        if (binary()) {
          for (auto rowOffset = offset_[colIdx];
               rowOffset < offset_[colIdx + 1]; ++rowOffset) {
            auto rowIdx = index_[rowOffset];
            if (rowRange.contains(rowIdx)) {
              y[rowIdx] += xAtCol;
            }
          }
        } else {
          for (auto rowOffset = offset_[colIdx];
               rowOffset < offset_[colIdx + 1]; ++rowOffset) {
            auto rowIdx = index_[rowOffset];
            if (rowRange.contains(rowIdx)) {
              y[rowIdx] += xAtCol * value_[rowOffset];
            }
          }
        }
      }
    }
  }

  template <typename W> void templateTimes(const W *x, W *y) {
    SizeR rowRange(0, rows());
    int numThreads = FLAGS_num_threads;
    assert(numThreads > 0);
    ThreadPool pool(numThreads);
    int numTasks = rowMajor() ? numThreads * 10 : numThreads;
    for (int i = 0; i < numTasks; ++i) {
      pool.add([this, x, y, rowRange, numTasks, i]() {
        rangeTimes(rowRange.evenDivide(numTasks, i), x, y);
      });
    }
    pool.startWorkers();
  }

  DArray<size_t> offset_;
  DArray<I> index_;
};

template <typename I, typename V>
MatrixPtr<V> SparseMatrix<I, V>::dotTimes(const MatrixPtr<V> &B) const {
  auto C = std::static_pointer_cast<const SparseMatrix<I, V>>(B);
  assert(rows() == C->rows());
  assert(cols() == C->cols());
  assert(nnz() == C->nnz());

  DArray<V> dot;
  if (binary()) {
    dot = C->value_;
  } else if (C->binary()) {
    dot = value_;
  } else {
    dot.resize(value_.size());
    for (int i = 0; i < value_.size(); ++i) {
      dot[i] = value_[i] * C->value_[i];
    }
  }
  return MatrixPtr<V>(new SparseMatrix<I, V>(info_, offset_, index_, dot));
}

// TODO: implement better design here, possible bug!!!
template <typename I, typename V>
MatrixPtr<V> SparseMatrix<I, V>::colBlock(SizeR range) const {
  assert(range.valid());
  if (rowMajor()) {
    assert(range.size() == cols());
    LOG(INFO) << "For row major matrix, only support return whole matrix";
    return MatrixPtr<V>(new SparseMatrix<I, V>(info_, offset_, index_, value_));
  } else {
    auto blkOffset = offset_.segment(SizeR(range.begin(), range.end() + 1));
    auto blkInfo = info_;
    range.to(blkInfo.mutable_col());
    blkInfo.set_nnz(blkOffset.back() - blkOffset.front());
    return MatrixPtr<V>(
        new SparseMatrix<I, V>(blkInfo, blkOffset, index_, value_));
  }
}

template <typename I, typename V>
MatrixPtr<V> SparseMatrix<I, V>::rowBlock(SizeR range) const {
  assert(range.valid());
  if (colMajor()) {
    assert(range.size() == rows());
    LOG(INFO) << "Limited support for sparse matrix operation";
    return MatrixPtr<V>(new SparseMatrix<I, V>(info_, offset_, index_, value_));
  } else {
    auto blkOffset = offset_.segment(SizeR(range.begin(), range.end() + 1));
    auto blkInfo = info_;
    range.to(blkInfo.mutable_row());
    blkInfo.set_nnz(blkOffset.back() - blkOffset.front());
    return MatrixPtr<V>(
        new SparseMatrix<I, V>(blkInfo, blkOffset, index_, value_));
  }
}

template <typename I, typename V>
MatrixPtr<V> SparseMatrix<I, V>::alterStorage() {
  assert(!empty());
  size_t innerSize = this->innerSize();
  size_t outerSize = this->outerSize();
  if (innerSize >= (size_t)kuint32max) {
    LOG(ERROR) << "Please run localize at first";
  }

  DArray<size_t> newOffset(innerSize + 1);
  newOffset.setZero();
  int numThreads = FLAGS_num_threads;
  assert(numThreads > 0);
  {
    ThreadPool pool(numThreads);
    for (int i = 0; i < numThreads; ++i) {
      SizeR range = SizeR(0, innerSize).evenDivide(numThreads, i);
      pool.add([this, range, &newOffset]() {
        for (I k : index_) {
          if (range.contains(k)) {
            ++newOffset[k + 1];
          }
        }
      });
    }
    pool.startWorkers();
  }
  for (int i = 0; i < innerSize; ++i) {
    newOffset[i + 1] += newOffset[i];
  }
  DArray<I> newIndex;
  DArray<V> newValue;
  {
    ThreadPool pool(numThreads);
    for (int i = 0; i < numThreads; ++i) {
      SizeR range = SizeR(0, innerSize).evenDivide(numThreads, i);
      pool.add([this, range, outerSize, &newOffset, &newIndex, &newValue]() {
        for (int i = 0; i < outerSize; ++i) {
          if (offset_[i] == offset_[i + 1]) {
            continue;
          }
          for (auto idx = offset_[i]; idx < offset_[i + 1]; ++idx) {
            auto ptr = index_[idx];
            if (range.contains(ptr)) {
              newIndex[newOffset[ptr]] = static_cast<I>(i);
              newValue[newOffset[ptr]] = value_[idx];
              ++newOffset[ptr];
            }
          }
        }
      });
    }
    pool.startWorkers();
  }
  for (int i = innerSize - 1; i > 0; --i) {
    newOffset[i] = newOffset[i - 1];
  }
  newOffset[0] = 0;

  auto newInfo = info_;
  newInfo.set_row_major(!info_.row_major());
  return MatrixPtr<V>(
      new SparseMatrix<I, V>(newInfo, newOffset, newIndex, newValue));
}

template <typename I, typename V>
std::ostream &operator<<(std::ostream &os, const SparseMatrix<I, V> &mat) {
  os << mat.debugString();
  return os;
}

template <typename I, typename V>
std::string SparseMatrix<I, V>::debugString() const {
  std::stringstream ss;
  int nnz = offset_.back() - offset_.front();
  ss << info_.DebugString() << std::endl
     << "offset: " << offset_ << std::endl
     << "index:  " << dbgstr(index_.data() + offset_[0], nnz) << std::endl;
  if (!binary()) {
    ss << "value: " << dbgstr(value_.data() + offset_[0], nnz) << std::endl;
  }
  return ss.str();
}
} // namespace mltools

#endif // __SPARSE_MATRIX_H__
