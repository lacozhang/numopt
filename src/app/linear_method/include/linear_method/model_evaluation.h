/*
 * =====================================================================================
 *
 *       Filename:  model_evaluation.h
 *
 *    Description:  evaluation code
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:02:47
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once

#include "data/stream_reader.h"
#include "proto/linear.pb.h"
#include "system/customer.h"
#include "system/sysutil.h"
#include "util/evaluation.h"

namespace mltools {
namespace linear {
class ModelEvaluation : public App {
public:
  ModelEvaluation(const Config &conf) : App(), conf_(conf) {}
  virtual ~ModelEvaluation() {}
  virtual void run() override;

private:
  typedef float Real;
  Config conf_;
};

void ModelEvaluation::run() {
  if (!IsScheduler()) {
    return;
  }
  std::unordered_map<Key, Real> weight;
  auto model = searchFiles(conf_.model_input());
  NOTICE("find %d model files", model.file_size());

  for (int i = 0; i < model.file_size(); ++i) {
    std::ifstream src(model.file(i));

    while (src.good()) {
      Key k;
      Real v;
      src >> k >> v;
      weight[k] = v;
    }
  }

  NOTICE("load %ld model entries", weight.size());
  auto data = searchFiles(conf_.validation_data());
  data.set_ignore_feature_group(true);
  NOTICE("find %d data files", data.file_size());

  DArray<Real> label;
  DArray<Real> predict;
  MatrixPtrList<Real> mat;
  StreamReader<Real> reader(data);
  bool good = false;

  do {
    good = reader.readMatrices(10000, &mat);
    CHECK_EQ(mat.size(), 2);
    label.append(mat[0]->value());

    DArray<Real> Xw(mat[1]->rows());
    Xw.setZero();
    auto X = std::static_pointer_cast<SparseMatrix<Key, Real>>(mat[1]);
    for (int i = 0; i < X->rows(); ++i) {
      Real re = 0;
      for (size_t j = X->offset()[i]; j < X->offset()[i + 1]; ++j) {
        auto it = weight.find(X->index()[j]);
        if (it != weight.end()) {
          re += it->second * (X->binary() ? 1 : X->value()[j]);
        }
      }
      Xw[i] = re;
    }
    predict.append(Xw);
    printf("\r                                             \r");
    printf("  load %lu examples", label.size());
    fflush(stdout);
  } while (good);
  printf("\n");

  NOTICE("auc: %f", Evaluation<Real>::auc(label, predict));
  NOTICE("accuracy: %f", Evaluation<Real>::accuracy(label, predict));
  NOTICE("logloss: %f", Evaluation<Real>::logloss(label, predict));
}
} // namespace linear
} // namespace mltools
