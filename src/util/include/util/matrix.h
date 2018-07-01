/*
 * =====================================================================================
 *
 *       Filename:  matrix.h
 *
 *    Description:  Abstract interface to matrix operation
 *
 *        Version:  1.0
 *        Created:  06/24/2018 20:27:04
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "proto/matrix.pb.h"
#include "util/dynamic_array.h"
#include "util/range.h"
#include <Eigen/Dense>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace mltools {

template <typename V> class Matrix;
template <typename V> using MatrixPtr = std::shared_ptr<Matrix<V>>;
template <typename V> using MatrixPtrList = std::vector<MatrixPtr<V>>;
template <typename V>
using MatrixPtrInitList = std::initializer_list<MatrixPtrList<V>>;

#define USING_MATRIX                                                           \
  using Matrix<V>::rows;                                                       \
  using Matrix<V>::cols;                                                       \
  using Matrix<V>::nnz;                                                        \
  using Matrix<V>::info_;                                                      \
  using Matrix<V>::value_;                                                     \
  using Matrix<V>::rowMajor;                                                   \
  using Matrix<V>::colMajor;                                                   \
  using Matrix<V>::empty;                                                      \
  using Matrix<V>::innerSize;                                                  \
  using Matrix<V>::outerSize;

template <typename V> class Matrix {
  pubic : Matrix() {}
  explicit Matrix(const MatrixInfo &info) : info_(info) {}
  Matrix(const MatrixInfo &info, const DArray<V> &value)
      : info_(info), value_(value) {}

  virtual void resize(size_t rows, size_t cols, size_t nnz = 0,
                      bool row_major = true) = 0;

  typedef Eigen::Matrix<V, Eigen::Dynamic, 1> EVec;
  EVec operator*(const Eigen::Ref<const EVec> &x) const { return times(x); }

  /**
   * @brief Return y = W * x
   *
   * return the result of Matrix vector product
   * @param x
   * @return y
   */
  EVec times(const Eigen::Ref<const EVec> &x) const {
    assert(x.size() == cols());
    EVec y(rows());
    times(x.data(), y.data());
    return y;
  }

  virtual void times(const V *x, V *y) const = 0;

  EVec transTimes(const Eigen::Ref<const EVec> &x) const {
    return trans()->times(x);
  }

  /// @brief Return element-wise product result
  virtual MatrixPtr<V> dotTimes(const MatrixPtr<V> &B) const = 0;

  /// @brief Return the Transpose of current matrix
  virtual MatrixPtr<V> trans() const = 0;

  MatrixPtr<V> toRowMajor() {
    return (rowMajor() ? MatrixPtr(this, [](Matrix<V> *p) {}) : alterStorage());
  }

  MatrixPtr<V> toColMajor() {
    return (colMajor() ? MatrixPtr(this, [](Matrix<V> *p) {}) : alterStorage());
  }

  virtual MatrixPtr<V> alterStorage() = 0;

  virtual MatrixPtr<V> rowBlock(SizeR range) const = 0;
  virtual MatrixPtr<V> colBlock(SizeR range) const = 0;

  virtual bool writeToBinFile(std::string filename) const = 0;

  virtual size_t memSize() { return value_.size() * sizeof(V); }

  const MatrixInfo &info() const { return info_; }
  MatrixInfo &info() { return info_; }
  uint64 rows() const { return info_.row().end() - info_.row().begin(); }
  uint64 cols() const { return info_.col().end() - info_.col().begin(); }
  uint64 nnz() const { return info_.nnz(); }

  bool rowMajor() const { return info_.row_major(); }

  bool colMajor() const { return !info_.row_major(); }

  size_t innerSize() const { return rowMajor() ? cols() : rows(); }

  size_t outerSize() const { return rowMajor() ? rows() : cols(); }

  bool empty() const { return (rows() == 0) && (cols() == 0); }

  DArray<V> value() const { return value_; }

  void tranposeInfo() {
    auto info = info_;
    *info_.mutable_row() = info.col();
    *info_.mutable_col() = info.row();
  }

  typedef Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      EMat;
  Eigen::Map<EMat> eigenMatrix() {
    assert(rowMajor());
    assert(info_.type() == MatrixInfo::DENSE);
    return Eigen::Map<EMat>(value_.data(), rows(), cols());
  }

  typedef Eigen::Array<V, Eigen::Dynamic, 1> EArr;
  Eigen::Map<EArr> eigenArray() { return value_.eigenArray(); }

  virtual std::string debugString() const { return info_.DebugString(); }

protected:
  MatrixInfo info_;
  DArray<V> value_;
};
} // namespace mltools

#endif // __MATRIX_H__
