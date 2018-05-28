#include "vectorsum.h"
namespace NNModel {

    void VectorSumLayer::Forward(const std::vector<boost::shared_ptr<RealVector>>& input, boost::shared_ptr<RealVector>& output) {
        if (input.size() != inputsize_) {
            NNForbidOperationMsg("Input size not expected");
        }

        for (int i = 0; i < inputsize_; ++i) {
            *output_ += *input[i];
        }
        output = output_;
    }

    void VectorSumLayer::InputGrad(const std::vector<boost::shared_ptr<RealVector>>& input, const boost::shared_ptr<RealVector>& gradin, boost::shared_ptr<RealVector>& gradout) {
        gradout = gradin;
    }

}