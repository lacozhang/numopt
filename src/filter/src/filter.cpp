/*
 * =====================================================================================
 *
 *       Filename:  filter.cpp
 *
 *    Description:  implementation of filter.h
 *
 *        Version:  1.0
 *        Created:  07/19/2018 08:55:48
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "filter/filter.h"
#include "filter/add_noise.h"
#include "filter/compressing.h"
#include "filter/fixing_float.h"
#include "filter/frequency_filter.h"
#include "filter/key_caching.h"
#include "filter/sparse_filter.h"

namespace mltools {
Filter *Filter::create(const mltools::FilterConfig &conf) {
  switch (conf.type()) {
  case FilterConfig::KEY_CACHING:
    return new KeyCachingFilter();
    break;
  case FilterConfig::COMPRESSING:
    return new CompressingFilter();
    break;
  case FilterConfig::FIXING_FLOAT:
    return new FixingFloatFilter();
    break;
  case FilterConfig::NOISE:
    return new AddNoiseFilter();
  default:
    CHECK(false) << "unknown filter type";
  }
}

FilterConfig *Filter::find(FilterConfig::Type type, mltools::Task *task) {
  for (int i = 0; i < task->filter_size(); ++i) {
    if (task->filter(i).type() == type) {
      return task->mutable_filter(i);
    }
  }
  return nullptr;
}
} // namespace mltools
