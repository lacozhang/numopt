#pragma once

#include "nnmodule.h"

#ifndef __TEXT_POOLING_H__
#define __TEXT_POOLING_H__

namespace NNModel {
    class TextMaxPoolingLayer : public NNLayerBase<RowRealMatrix, RealVector, RealVector, RowRealMatrix> {
    public:

        typedef NNLayerBase<RowRealMatrix, RealVector, RealVector, RowRealMatrix> BaseType;

        TextMaxPoolingLayer(int stack, int convsize) {
            stack_ = stack;
            convsize_ = convsize;
            output_ = boost::make_shared<RealVector>();
            inputgrad_ = boost::make_shared<RowRealMatrix>();
            output_->resize(convsize*stack);
            poolingidx_.resize(convsize*stack);
        }
        ~TextMaxPoolingLayer(){}

        void Forward(const RowRealMatrix& input, boost::shared_ptr<RealVector>& output);
        void Backward(const RowRealMatrix& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RowRealMatrix>& gradout) {
            InputGrad(input, gradin, gradout);
        }
        void Backward(const RowRealMatrix& input, const boost::shared_ptr<RealVector>& gradin) {
            BaseType::Backward(input, gradin);
        }
        void ResetParamGrad() {}

    protected:
        void ParamGrad(const RowRealMatrix& input, const boost::shared_ptr<RealVector>& gradin) {
            BaseType::ParamGrad(input, gradin);
        }
        void InputGrad(const RowRealMatrix& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RowRealMatrix>& gradout);

    private:
        int stack_, convsize_;
        Eigen::VectorXi poolingidx_;
    };
}


#endif // __TEXT_POOLING_H__