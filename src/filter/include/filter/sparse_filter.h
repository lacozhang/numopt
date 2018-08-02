/*
 * =====================================================================================
 *
 *       Filename:  sparse_filter.h
 *
 *    Description:  sparse filter
 *
 *        Version:  1.0
 *        Created:  07/22/2018 15:54:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "filter/filter.h"
namespace mltools {
class SparseFilter : public Filter {
public:
  SparseFilter() {
    // special mark get every bit 1
    memcpy(&doubleV_, &kuint64max, sizeof(double));
    memcpy(&floatV_, &kuint32max, sizeof(float));
  }

  void mark(float *v) { *v = floatV_; }

  void mark(double *v) { *v = doubleV_; }

  bool marked(double v) { return v != doubleV_; }

  bool marked(float v) { return v != floatV_; }

private:
  float floatV_;
  double doubleV_;
};
} // namespace mltools
