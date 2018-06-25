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
#include <initializer_list>
#include <memory>
#include <vector>

namespace mltools {

template <typename V> class Matrix;
template <typename V> using MatrixPtr = std::shared_ptr<Matrix<V>>;
template <typename V> using MatrixPtrList = std::vector<MatrixPtr<V>>;
template <typename V>
using MatrixPtrInitList = std::initializer_list<MatrixPtrList<V>>;

#define USING_MATRIX                                                           \
  using Matrix<V>::rows;                                                       \
  using Matrix<V>::cols;

} // namespace mltools
