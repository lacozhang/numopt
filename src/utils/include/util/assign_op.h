/*
 * =====================================================================================
 *
 *       Filename:  assign_op.h
 *
 *    Description:  operator
 *
 *        Version:  1.0
 *        Created:  07/24/2018 19:44:26
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "proto/assign_op.pb.h"
#include <glog/logging.h>

namespace mltools {

/// @brief template function for right op= left
template <typename T>
T &AssignOp(T &right, const T &left, const AssignOpType &op) {
  switch (op) {
  case AssignOpType::ASSIGN:
    right = left;
    break;
  case AssignOpType::PLUS:
    right += left;
    break;
  case AssignOpType::MINUS:
    right -= left;
    break;
  case AssignOpType::TIMES:
    right *= left;
    break;
  case AssignOpType::DIVIDE:
    right /= left;
    break;
  default:
    CHECK(false) << "Not supported yet";
    break;
  }
  return right;
}

template <typename T>
T &AssignOpI(T &right, const T &left, const AssignOpType &op) {
  switch (op) {
  case AssignOpType::ASSIGN:
    right = left;
    break;
  case AssignOpType::PLUS:
    right += left;
    break;
  case AssignOpType::MINUS:
    right -= left;
    break;
  case AssignOpType::TIMES:
    right *= left;
    break;
  case AssignOpType::DIVIDE:
    right /= left;
    break;
  case AssignOpType::AND:
    right &= left;
    break;
  case AssignOpType::OR:
    right |= left;
    break;
  case AssignOpType::XOR:
    right ^= left;
    break;
  }
  return right;
}
} // namespace mltools
