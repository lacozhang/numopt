#pragma once
#include <cmath>
#include "nnmodule.h"
#ifndef __ACTIVATE_LAYER_H__
#define __ACTIVATE_LAYER_H__

namespace NNModel {

    enum class ActivateType {
        Sigmoid,
        HardTanh,
        ReLU
    };

    class ActivateBase {
    public:
        ActivateBase() {}
        ~ActivateBase() {}

        virtual double activate(double input) = 0;
        virtual double derivate(double input, double output, double grad) = 0;
    };

    class SigmoidAcitvate : public ActivateBase {
    public:
        virtual double activate(double input) override {
            return 1.0 / (1.0 + std::exp(-input));
        }
        virtual double derivate(double input, double output, double grad) override {
            return grad * (1 - output)*output;
        }
    };

    class HardTanhActivate : public ActivateBase {
    public:
        HardTanhActivate(double minv=-1, double maxv=1) : minval_(minv), maxval_(maxv) {

        }
        virtual double activate(double input) override {
            return std::min(std::max(input, minval_), maxval_);
        }
        virtual double derivate(double input, double output, double grad) override {
            return grad*(input >= minval_ && input <= maxval_ ? 1 : 0);
        }

    private:
        double minval_, maxval_;
    };

    class ReluActivate : public ActivateBase {
    public:
        virtual double activate(double input) override {
            return input >= 0 ? input : 0;
        }
        virtual double derivate(double input, double output, double grad) override {
            return output >= 0 ? 1 : 0;
        }
    };

    class ActivateLayer : public NNLayerBase<RealVector, RealVector, RealVector, RealVector> {
    public:
        ActivateLayer(ActivateType type);

        void Forward(const RealVector& input, boost::shared_ptr<RealVector>& output);
        void Backward(const RealVector& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RealVector>& gradout) {
            InputGrad(input, gradin, gradout);
        }
        void Backward(const RealVector& input, const boost::shared_ptr<RealVector>& gradin) {
            NNForbidOperation;
        }
        void ResetParamGrad() {}

    protected:
        void ParamGrad(const RealVector& input, const boost::shared_ptr<RealVector>& gradin) {
            NNForbidOperation;
        }
        void InputGrad(const RealVector& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RealVector>& gradout);

    private:
        boost::shared_ptr<ActivateBase> func_;
    };

}


#endif // !__ACTIVATE_LAYER_H__
