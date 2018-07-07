/*
 * =====================================================================================
 *
 *       Filename:  dense_matrix.h
 *
 *    Description:  inherit the matrix interface
 *
 *        Version:  1.0
 *        Created:  06/25/2018 20:45:54
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
#include "util/matrix.h"

#ifndef __DENSE_MATRIX_H__
#define __DENSE_MATRIX_H__

namespace mltools {

template <typename V> class DenseMatrix : public Matrix<V> {
public:
  USING_MATRIX;
  DenseMatrix() {}
  DenseMatrix(size_t rows, size_t cols, bool rowMajor = true) {
    resize(rows, cols, rows * cols, rowMajor);
  }
  DenseMatrix(const MatrixInfo &info, const DArray<V> &value)
      : Matrix<V>(info, value) {}
  virtual ~DenseMatrix() {}

  void resize(size_t rows, size_t cols, size_t nnz, bool rowMajor) override;

  /// TODO: implement this
  virtual void times(const V *x, V *y) const override { assert(false); }

  /// @brief Not implemented
  virtual MatrixPtr<V> dotTimes(const MatrixPtr<V> &B) const override {
    assert(false);
    return MatrixPtr<V>();
  }

  virtual MatrixPtr<V> trans() const override {
    assert(false);
    return MatrixPtr<V>();
  }

  virtual MatrixPtr<V> alterStorage() const override;

  virtual MatrixPtr<V> rowBlock(SizeR range) const override {
    if (colMajor()) {
      assert(range == SizeR(0, rows()));
    }
    auto info = info_;
    range.to(info.mutable_row());
    info.set_nnz(range.size() * cols());
    return MatrixPtr<V>(new DenseMatrix(info, value_.segment(range * cols())));
  }

  virtual MatrixPtr<V> colBlock(SizeR range) const override {
    if (rowMajor()) {
      assert(range == SizeR(0, cols()));
      LOG(FATAL) << "Can't extract several columns for row major matrix";
    }
    auto info = info_;
    range.to(info.mutable_col());
    info.set_nnz(range.size() * rows());
    return MatrixPtr<V>(new DenseMatrix(info, value_.segment(range * rows())));
  }

  virtual bool writeToBinFile(std::string filepath) const override {
    return writeProtoToASCIIFile(info_, filepath + ".info") &&
           value_.writeToFile(filepath + ".value");
  }

  virtual std::string debugString() const override {
    std::stringstream ss;
    ss << rows() << " x " << cols() << " dense matrix " << std::endl;
    ss << dbgstr(value_.data(), value_.size(), 8);
    return ss.str();
  }
};

template <typename V>
void DenseMatrix<V>::resize(size_t rows, size_t cols, size_t nnz,
                            bool rowMajor) {
  info_.set_type(MatrixInfo::DENSE);
  info_.set_row_major(rowMajor);
  SizeR(0, rows).to(info_.mutable_row());
  SizeR(0, cols).to(info_.mutable_col());
  nnz = rows * cols;

  info_.set_nnz(nnz);
  info_.set_sizeof_val(sizeof(V));

  value_.resize(nnz);
  value_.setValue();
}

template <typename V> MatrixPtr<V> DenseMatrix<V>::alterStorage() const {
  auto inner = innerSize(), outer = outerSize();
  assert(value_.size() == inner * outer);
  DArray<V> newArray(value_.size());
  for (size_t i = 0; i < inner; ++i) {
    for (size_t j = 0; j < outer; ++j) {
      newArray[i * outer + j] = value_[j * inner + i];
    }
  }

  auto newInfo = info_;
  newInfo.set_row_major(!info_.row_major());
  return MatrixPtr<V>(new DenseMatrix<V>(newInfo, newArray));
}
} // namespace mltools

#endif // __DENSE_MATRIX_H__
