/*
 * =====================================================================================
 *
 *       Filename:  penalty.h
 *
 *    Description:  regularization code
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:03:02
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/linear.pb.h"
#include "util/common.h"
#include "util/matrix.h"

namespace mltools {
namespace linear {

/// @brief main interface to penalty
template <typename T> class Penalty {
public:
  Penalty() {}
  virtual ~Penalty() {}

  /// @brief evaluate the objective
  virtual T eval(const Matrix<T> &model) = 0;

  /**
   * @brief Solve the proximal operator
   *
   * \f$ \argmin_x 0.5/\eta (x - z)^2 + h(x)\f$, where h denote this penatly,
   * and in proximal gradient descent, z = w - eta * grad
   *
   * @param z
   * @param eta
   * @return
   */
  virtual T proximal(T z, T eta) = 0;
};

template <typename T> class ElasticNet : public Penalty<T> {
public:
  ElasticNet(T lambda1, T lambda2) : lambda1_(lambda1), lambda2_(lambda2) {
    CHECK_GE(lambda1, 0);
    CHECK_GE(lambda2, 0);
  }

  ~ElasticNet() {}

  virtual T eval(const MatrixPtr<T> &model) override { return 0; }

  T proximal(T z, T eta) override {
    CHECK_GT(eta, 0);
    T leta = lambda1_ * eta;
    if (z <= leta && z >= -leta)
      return 0;
    return z > 0 ? (z - leta) / (1 + lambda2_ * eta)
                 : (z + leta) / (1 + lambda2_ * eta);
  }

private:
  T lambda1_, lambda2_;
};
} // namespace linear
} // namespace mltools
