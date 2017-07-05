#include "dropout.h"

namespace NNModel {


    DropoutLayer::DropoutLayer(double rate, int size) : rndgen_(0.0, 1.0) {
        size_ = size;
        rate_ = rate;
        gatestate_.resize(size);
        output_ = boost::make_shared<RealVector>();
        inputgrad_ = boost::make_shared<RealVector>();
        output_->resize(size);
        inputgrad_->resize(size);
    }


    void DropoutLayer::Forward(const RealVector& input, boost::shared_ptr<RealVector>& output) {
        if (trainstate_) {
            output_->setZero();
            for (int i = 0; i < size_; ++i) {
                double prob = rndgen_(engine_);
                if (prob < rate_) {
                    gatestate_[i] = true;
                    output_->coeffRef(i) = input.coeff(i);
                }
                else {
                    gatestate_[i] = false;
                }
            }
        }
        else {
            *output_ = input;
            *output_ *= rate_;
        }
    }


    void DropoutLayer::InputGrad(const RealVector& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RealVector>& gradout) {
        inputgrad_->setZero();
        if (gradin->size() != inputgrad_->size()) {
            NNForbidOperationMsg("Gradient size error");
        }
        for (int i = 0; i < size_; ++i)
            if (gatestate_[i]) inputgrad_->coeffRef(i) = gradin->coeff(i);
    }
}