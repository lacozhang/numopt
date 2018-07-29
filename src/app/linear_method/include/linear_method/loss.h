/*
 * =====================================================================================
 *
 *       Filename:  loss.h
 *
 *    Description:  loss function used
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:02:23
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
#include <Eigen/Dense>

namespace mltools {
namespace linear {

template <typename T> class Loss {
public:
  /// @brief evaluate the loss value
  virtual T evaluate(const MatrixPtrList<T> &data) = 0;

  /// @brief compute the gradients of the data
  virtual void compute(const MatrixPtrList<T> &data,
                       MatrixPtrList<T> gradients) = 0;
};

template <typename T> class ScalarLoss : public Loss<T> {
public:
  typedef Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> EArray;

  /// @brief keep undefined.
  virtual T evaluate(const EArray &y, const EArray &Xw) = 0;

  virtual void compute(const EArray &y, const MatrixPtr<T> &X, const EArray &Xw,
                       EArray gradient, EArray diagHession) = 0;

  virtual T evaluate(const MatrixPtrList<T> &data) override {
    CHECK_EQ(data.size(), 2);
    DArray<T> y(data[0]->value());
    DArray<T> Xw(data[1]->value());
    CHECK_EQ(y.size(), Xw.size());
    return evaluate(y.eigenArray(), Xw.eigenArray());
  }

  virtual void compute(const MatrixPtrList<T> &data,
                       MatrixPtrList<T> gradients) override {
    if (gradients.size() == 0) {
      return;
    }

    CHECK_EQ(data.size(), 3);
    auto y = data[0]->value();
    auto X = data[1];
    auto Xw = data[2]->value();

    CHECK_EQ(y.size(), Xw.size());
    CHECK_EQ(y.size(), X->rows());

    CHECK(gradients[0]);
    auto gradient = gradients[0]->value();
    auto diagHession = gradients.size() > 1 && gradients[1]
                           ? gradients[1]->value()
                           : DArray<T>();
    if (gradient.size() != 0) {
      CHECK_EQ(gradient.size(), X->cols());
    }
    if (diagHession.size() != 0) {
      CHECK_EQ(diagHession.size(), X->cols());
    }

    if (!y.size()) {
      return;
    }
    compute(y.eigenArray(), X, Xw.eigenArray(), gradient.eigenArray(),
            diagHession.eigenArray());
  }
};

template <typename T> class BinaryClassificationLoss : public ScalarLoss<T> {};

template <typename T> class LogitLoss : public BinaryClassificationLoss<T> {
public:
  typedef Eigen::Array<T, Eigen::Dynamic, 1> EArray;
  typedef Eigen::Map<EArray> EArrayMap;

  virtual T evaluate(const EArrayMap &y, const EArrayMap &Xw) override {
    return log(1 + exp(-y * Xw)).sum();
  }

  virtual void compute(const EArrayMap &y, const MatrixPtr<T> &X,
                       const EArrayMap &Xw, EArrayMap gradient,
                       EArrayMap diagHession) override {
    EArray tau = 1 / (1 + exp(y * Xw));
    if (gradient.size()) {
      gradient = X->transTimes(-y * tau);
    }

    if (diagHession.size()) {
      diagHession = X->dotTimes(X)->transTimes(tau * (1 - tau));
    }
  }
};

template <typename T>
class SquareHingeLoss : public BinaryClassificationLoss<T> {
public:
  typedef Eigen::Array<T, Eigen::Dynamic, 1> EArray;
  typedef Eigen::Map<EArray> EArrayMap;

  virtual T evaluate(const EArrayMap &y, const EArrayMap &Xw) override {
    return (1 - y * Xw).max(EArray::Zero(y.size())).square().sum();
  }

  virtual void compute(const EArrayMap &y, const MatrixPtr<T> &X,
                       const EArrayMap &Xw, EArrayMap gradient,
                       EArrayMap diagHession) {
    gradient = -2 * X->transTimes(y * (y * Xw > 1.0).template cast<T>());
  }
};

template <typename T> using LossPtr = std::shared_ptr<Loss<T>>;

template <typename T> static LossPtr<T> createLoss(const LossConfig &config) {
  switch (config.type()) {
  case LossConfig::LOGIT:
    return LossPtr<T>(new LogitLoss<T>());
  case LossConfig::SQUARE_HINGE:
    return LossPtr<T>(new LogitLoss<T>());
  default:
    CHECK(false) << "unknown type : " << config.DebugString();
  }

  return LossPtr<T>(nullptr);
}
} // namespace linear
} // namespace mltools
