/*
 * =====================================================================================
 *
 *       Filename:  kv_vector.h
 *
 *    Description:  value is vector; distributed representation
 *
 *        Version:  1.0
 *        Created:  07/23/2018 21:09:50
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once
#include "system/sysutil.h"
#include "parameter/parameter.h"
#include "filter/frequency_filter.h"
#include "util/parallel_ordered_match.h"

namespace mltools {
  
  /**
   * @brief key-value vectors
   *  keys of type K, value is a fixed length array of type V. Physical stroage format:
   *  key_0,  ... key_n
   *  val_00, ... val_n0
   *  val_01, ... val_n1
   *   ...    ... ...
   *  val_0k, ... val_nk
   *  keys are ordered and unique. values stored in a column-major format. support multiple channels.
   */
  template <typename K, typename V>
  class KVVector : public Parameter {
  public:
  };
}
