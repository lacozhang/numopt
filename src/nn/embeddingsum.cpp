#include "embeddingsum.h"

namespace NNModel {

    EmbeddingSumLayer::EmbeddingSumLayer(double * param, double * grad, int vocabsize, int embeddingsize) :
        param_(param, vocabsize, embeddingsize), grad_(grad, vocabsize, embeddingsize) {
        vocabsize_ = vocabsize;
        embedsize_ = embeddingsize;
        output_ = boost::make_shared<RealVector>();
        output_->resize(embedsize_);
    }
    void EmbeddingSumLayer::Forward(const DataSamples & input, boost::shared_ptr<RealVector>& output) {
        if (!output_) {
            NNForbidOperationMsg("Output empty");
        }
        *output_ = (input.template cast<double>() * param_).colwise().sum();
        output = output_;
    }


    void EmbeddingSumLayer::ParamGrad(const DataSamples & input, const boost::shared_ptr<RealVector>& gradin){
        for (int i = 0; i < input.outerSize(); ++i) {
            for (auto iter = DataSamples::InnerIterator(input, i); iter; ++iter) {
                grad_.row(iter.col()) += iter.value() * *gradin;
            }
        }
    }
}