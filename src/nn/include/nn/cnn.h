#pragma once
#include "dataop/vocabulary.h"
#include "model/AbstractModel.h"
#include "nn/activatelayer.h"
#include "nn/crossentropy.h"
#include "nn/dropout.h"
#include "nn/embedding.h"
#include "nn/embeddingsum.h"
#include "nn/linearlayer.h"
#include "nn/textconv.h"
#include "nn/textpooling.h"
#include "nn/vectorsum.h"
#include "nnquery.h"
#include <boost/shared_ptr.hpp>

#ifndef __CNN_H__
#define __CNN_H__

namespace NNModel {

class CNNModel : public AbstractModel<RealVector, NNQueryFeature, NNQueryLabel,
                                      RealSparseVector, RealVector> {
public:
  typedef AbstractModel<DenseVector, DataSamples, LabelVector, SparseVector,
                        DenseVector>
      BaseModelType;

  CNNModel();
  ~CNNModel();

  virtual void InitFromCmd(int argc, const char *argv[]) override;
  virtual void InitFromData(DataIterator &iterator) override;
  virtual void Init() override;

  virtual RealVector &GetParameters() const override { return *param_; }

  virtual RealVector &GetParameters() override { return *param_; }

  size_t FeatureDimension() const override { return 0; }

  virtual bool LoadModel(std::string model) override;
  virtual bool SaveModel(std::string model) override;

  virtual double Learn(NNQueryFeature &samples, NNQueryLabel &labels,
                       RealSparseVector &grad) override;
  virtual double Learn(NNQueryFeature &samples, NNQueryLabel &labels,
                       RealVector &grad) override;

  virtual void Inference(NNQueryFeature &samples,
                         NNQueryLabel &labels) override;

  virtual double Evaluate(NNQueryFeature &samples, NNQueryLabel &labels,
                          std::string &summary) override;

private:
  static const char *kEmbeddingSizeOption, *kHiddenSizeOption,
      *kConvFilterSizeOptioin, *kConvWindowSizeOption, *kPoolingSizeOption,
      *kDropOutOption;
  boost::shared_ptr<Vocabulary> vocab_, label_;

  // parameters
  int vocabsize_, embedsize_, hiddensize_, convfilters_, convsize_, poolstack_,
      labelsize_;
  double dropout_;

  boost::shared_ptr<RealVector> param_, grad_;
  boost::shared_ptr<EmbeddingLayer> embedlayer_;
  boost::shared_ptr<TextConvLayer> convlayer_;
  boost::shared_ptr<TextMaxPoolingLayer> poolinglayer_;
  boost::shared_ptr<ActivateLayer> poolactlayer_;
  boost::shared_ptr<EmbeddingSumLayer> bowhiddenlayer_, bowoutputlayer_;
  boost::shared_ptr<VectorSumLayer> hiddensumlayer_, outputsumlayer_;
  boost::shared_ptr<LinearLayer> hiddenlayer_;
  boost::shared_ptr<LinearLayer> directlayer_;
  boost::shared_ptr<ActivateLayer> hiddenactlayer_;
  boost::shared_ptr<DropoutLayer> hiddendroplayer_;
  boost::shared_ptr<LinearLayer> outputlayer_;
  boost::shared_ptr<CrossEntropyLoss> losslayer_;
};

} // namespace NNModel

#endif // __CNN_H__
