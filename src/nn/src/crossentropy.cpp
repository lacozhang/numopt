#include "nn/crossentropy.h"

namespace NNModel {

    double CrossEntropyLoss::Forward(const RealVector& input, const int& golden) {
        if (input.size() != labelsize_) {
            NNForbidOperationMsg("Input size not correct");
        }

        double total = 0;
        softmax_ = input;
        softmax_.array() -= input.maxCoeff();
        for (int i = 0; i < labelsize_; ++i) {
            softmax_.coeffRef(i) = FAST_EXP(input.coeff(i));
        }
        total = softmax_.sum();

        softmax_ /= total;
        softmax_.array().log();
        return -1.0*softmax_.coeff(golden);
    }

    void CrossEntropyLoss::BackWard(const RealVector & input, const int & golden, boost::shared_ptr<RealVector>& grads) {
        grad_->setZero();
        *grad_ = softmax_;
        grad_->coeffRef(golden) -= 1.0;
        grads = grad_;
    }
}