#include "activatelayer.h"

namespace NNModel {

    ActivateLayer::ActivateLayer(ActivateType type) {
        switch (type)
        {
        case NNModel::ActivateType::Sigmoid:
            func_ = boost::make_shared<SigmoidAcitvate>();
            break;
        case NNModel::ActivateType::HardTanh:
            func_ = boost::make_shared<HardTanhActivate>();
            break;
        case NNModel::ActivateType::ReLU:
            func_ = boost::make_shared<ReluActivate>();
            break;
        default:
            break;
        }

        if (!func_) {
            BOOST_LOG_TRIVIAL(error) << "Allocate activate object failed";
            return;
        }

        output_ = boost::make_shared<RealVector>();
        inputgrad_ = boost::make_shared<RealVector>();
    }

    void ActivateLayer::Forward(const RealVector& input, boost::shared_ptr<RealVector>& output) {
        output_->resizeLike(input);
        output_->setZero();
        for (int i = 0; i < input.size(); ++i) {
            output_->coeffRef(i) = func_->activate(input.coeff(i));
        }

        output = output_;
    }


    void ActivateLayer::InputGrad(const RealVector& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RealVector>& gradout) {
        inputgrad_->resizeLike(input);
        inputgrad_->setZero();
        for (int i = 0; i < input.size(); ++i) {
            inputgrad_->coeffRef(i) = func_->derivate(input.coeff(i), output_->coeff(i), gradin->coeff(i));
        }
        gradout = inputgrad_;
    }
}