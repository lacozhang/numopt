/*
 * =====================================================================================
 *
 *       Filename:  add_nise.h
 *
 *    Description:  add noise filter
 *
 *        Version:  1.0
 *        Created:  07/22/2018 15:52:33
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
#include <random>


namespace mltools {
  
  class AddNoiseFilter : public Filter {
  public:
    void encode(Message *msg) {
      auto filterConf = CHECK_NOTNULL(find(FilterConfig::NOISE, msg));
      int n = msg->value_.size();
      CHECK_EQ(n, msg->task_.value_type_size());
      for(int i=0; i<n; ++i) {
        if(msg->value_[i].size() == 0) {
          continue;
        }
        auto type = msg->task_.value_type(i);
        if(type == DataType::FLOAT) {
          addNoise<float>(msg->value_[i], filterConf);
        }
        if(type == DataType::DOUBLE) {
          addNoise<double>(msg->value_[i], filterConf);
        }
      }
    }
    
  private:
    template <typename V>
    void addNoise(const DArray<char> &array, FilterConfig *cf) {
      std::default_random_engine generator;
      std::normal_distribution<V> distribution((V)cf->mean(), (V)cf->std());
      DArray<V> data(array);
      for(size_t i = 0; i<data.size(); ++i) {
        data[i] += distribution(generator);
      }
    }
  };
}
