/*
 * =====================================================================================
 *
 *       Filename:  evaluation.h
 *
 *    Description:  model evaluation
 *
 *        Version:  1.0
 *        Created:  07/29/2018 11:03:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "util/dynamic_array.h"
#include "util/parallel_sort.h"

namespace mltools {
  
  template <typename V>
  class Evaluation {
  public:
    static V auc(const DArray<V> &label,
                 const DArray<V> &predict);
    
    static V accuracy(const DArray<V> &label,
                      const DArray<V> &predict,
                      V threshold = 0);
    
    static V logloss(const DArray<V> &label,
                     const DArray<V> &predict);
  };
  
  template <typename V>
  V Evaluation<V>::auc(const DArray<V> &label, const DArray<V> &predict) {
    int n = label.size();
    CHECK_EQ(n, predict.size());
    struct Entry {
      V label_;
      V pred_;
    };
    DArray<Entry> buff(n);
    for(int i=0; i<n; ++i) {
      buff[i].label_ = label[i];
      buff[i].pred_ = predict[i];
    }
    std::sort(buff.data(), buff.data()+n, [](const Entry &a, const Entry &b){
      return a.pred_ < b.pred_;
    });
    V area = 0, cumTp = 0;
    for(int i=0; i<n; ++i) {
      if(buff[i].label_ > 0) {
        cumTp += 1;
      } else {
        area += cumTp;
      }
    }
    
    area /= cumTp*(n-cumTp);
    return area < 0.5 ? (1-area) : area;
  }
  
  template <typename V>
  V Evaluation<V>::accuracy(const DArray<V> &label,
                            const DArray<V> &predict,
                            V threshold) {
    int n = label.size();
    CHECK_EQ(n, predict.size());
    double tp = 0;
    for(int i=0; i<n; ++i) {
      tp += label[i] * (predict[i] - threshold) > 0 ? 1 : 0;
    }
    V acc = tp / (V)n;
    return acc > 0.5 ? acc: 1-acc;
  }
  
  template <typename V>
  V Evaluation<V>::logloss(const DArray<V> &label, const DArray<V> &predict) {
    int n = label.size();
    CHECK_EQ(n, predict.size());
    V loss = 0;
    for(int i=0; i<n; ++i) {
      V y = label[i] > 0;
      V p = 1 / (1 + std::exp(-predict[i]));
      loss += y * log(p) + (1-y)*log(1-p);
    }
    return - loss / n;
  }
}
