#pragma once

#ifndef __TEXTCONV_H__
#define __TEXTCONV_H__

#include "nnmodule.h"
namespace NNModel {
    class TextConvLayer : public NNLayerBase<RowRealMatrix, RowRealMatrix, RowRealMatrix, RowRealMatrix> {
    public:

        typedef NNLayerBase<RowRealMatrix, RowRealMatrix, RowRealMatrix, RowRealMatrix> BaseType;

        TextConvLayer(double* parambase, double* gradbase, int numfilters, int windowsize, int convsize, int stride);
        ~TextConvLayer() {}
        
        void Forward(const RowRealMatrix& input, boost::shared_ptr<RowRealMatrix>& output);
        void Backward(const RowRealMatrix& input, const boost::shared_ptr<RowRealMatrix>& gradin, boost::shared_ptr<RowRealMatrix>& gradout);

        void Backward(const RowRealMatrix& input, const boost::shared_ptr<RowRealMatrix>& gradin) {
            BaseType::Backward(input, gradin);
        }
        void ResetParamGrad() {
            grad_.setZero();
        }

    protected:
        void ParamGrad(const RowRealMatrix& input, const boost::shared_ptr<RowRealMatrix>& gradin) {
            NNForbidOperation;
        }
        void InputGrad(const RowRealMatrix& input, const boost::shared_ptr<RowRealMatrix>& gradin, boost::shared_ptr<RowRealMatrix>& gradout) {
            NNForbidOperation;
        }

    private:
        inline int outputrowsize(size_t inputrows) {
            int count = 0;
            for (int i = 0; i + windowsize_ <= inputrows; i += stride_) ++count;
            return count + 1;
        }

        Eigen::Map<RowRealMatrix> param_, grad_;
        int stride_, numfilters_, convsize_, windowsize_;
    };
}

#endif // !__TEXTCONV_H__
