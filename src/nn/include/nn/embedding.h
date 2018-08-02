#pragma once
#include "nnmodule.h"

#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__
namespace NNModel {

// for embedding layer, only need to udpate gradient of parameter

class EmbeddingLayer : public NNLayerBase<DataSamples, RowRealMatrix,
                                          RowRealMatrix, RowRealMatrix> {
public:
  EmbeddingLayer(double *parambase, double *gradbase, int vocab, int embedsize);
  ~EmbeddingLayer() {}

  void Forward(const DataSamples &input,
               boost::shared_ptr<RowRealMatrix> &output);

  void Backward(const DataSamples &input,
                const boost::shared_ptr<RowRealMatrix> &gradin,
                boost::shared_ptr<RowRealMatrix> &gradout) {
    NNForbidOperation;
  }

  void Backward(const DataSamples &input,
                const boost::shared_ptr<RowRealMatrix> &gradin);

  void ResetParamGrad() { grad_.setZero(); }

protected:
  void ParamGrad(const DataSamples &input,
                 const boost::shared_ptr<RowRealMatrix> &grad);
  void InputGrad(const DataSamples &input,
                 const boost::shared_ptr<RowRealMatrix> &gradin,
                 boost::shared_ptr<RowRealMatrix> &gradout) {
    NNForbidOperation;
  }

private:
  Eigen::Map<RowRealMatrix> param_;
  Eigen::Map<RowRealMatrix> grad_;
  int vocabsize_, embedsize_;
};
} // namespace NNModel

#endif // !__EMBEDDING_H__
