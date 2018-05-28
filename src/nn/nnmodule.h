#pragma once
#include <boost/log/trivial.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "../typedef.h"
#include "nnmacros.h"

#ifndef __NN_MODULE_H__
#define __NN_MODULE_H__

namespace NNModel {

    template<typename InputType, typename OutputType, typename GradInputType, typename GradOoutputType>
    class NNLayerBase {
    public:
        NNLayerBase() {}
        virtual ~NNLayerBase() {}

        // sparse input like embedding layer
        // input is sparse matrix, each row is a feature representation, output is a dense matrix
        void Forward(const InputType& input, boost::shared_ptr<OutputType>& output) {
            NNForbidOperation;
        }
        void Backward(const InputType& input, const boost::shared_ptr<GradInputType>& gradin, boost::shared_ptr<GradOoutputType>& gradout) {
            NNForbidOperation;
        }
        void Backward(const InputType& input, const boost::shared_ptr<GradInputType>& gradin) {
            NNForbidOperation;
        }
        void ResetParamGrad() {
            NNForbidOperation;
        }

        void SetInTrain() {
            trainstate_ = true;
        }

        void SetInTest() {
            trainstate_ = false;
        }

    protected:

        void ParamGrad(const InputType& input, const boost::shared_ptr<GradInputType>& gradin) {
            NNForbidOperation;
        }
        void InputGrad(const InputType& input, const boost::shared_ptr<GradInputType>& gradin, boost::shared_ptr<GradOoutputType>& gradout) {
            NNForbidOperation;
        }

        boost::shared_ptr<OutputType> output_;
        boost::shared_ptr<GradOoutputType> inputgrad_;
        bool trainstate_;
    };
}

#endif // !__NN_MODULE_H__
