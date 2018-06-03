#pragma once

#include "nnmodule.h"

#ifndef __EMBEDDING_SUM_H__
#define __EMBEDDING_SUM_H__

namespace NNModel {
    class EmbeddingSumLayer : public NNLayerBase<DataSamples, RealVector, RealVector, RowRealMatrix> {
    public:

        typedef NNLayerBase<DataSamples, RealVector, RealVector, RowRealMatrix> BaseType;

        EmbeddingSumLayer(double *param, double* grad, int vocabsize, int embeddingsize);
        ~EmbeddingSumLayer(){}

        void Forward(const DataSamples& input, boost::shared_ptr<RealVector>& output);

        void Backward(const DataSamples& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RowRealMatrix>& gradout) {
            BaseType::Backward(input, gradin, gradout);
        }

        void Backward(const DataSamples& input, const boost::shared_ptr<RealVector>& gradin) {
            ParamGrad(input, gradin);
        }
        void ResetParamGrad() {
            grad_.setZero();
        }

    protected:

        void ParamGrad(const DataSamples& input, const boost::shared_ptr<RealVector>& gradin);
        void InputGrad(const DataSamples& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RowRealMatrix>& gradout) {
            BaseType::InputGrad(input, gradin, gradout);
        }
    private:
        int vocabsize_, embedsize_;
        Eigen::Map<RowRealMatrix> param_, grad_;
    };
}

#endif // __EMBEDDING_SUM_H__