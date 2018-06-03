#include "nn/embedding.h"

namespace NNModel {

    EmbeddingLayer::EmbeddingLayer(double * parambase, double * gradbase, int vocab, int embedsize) :
        param_(parambase, vocab, embedsize),
        grad_(gradbase, vocab, embedsize) {
        embedsize_ = embedsize;
        vocabsize_ = vocab;
        output_ = boost::make_shared<RowRealMatrix>();
        inputgrad_.reset();
    }

    void EmbeddingLayer::Forward(const DataSamples& input, boost::shared_ptr<RowRealMatrix>& output) {
        output_->resize(input.rows(), embedsize_);
        output_->setZero();
        *output_ = input.template cast<double>()* param_;
        output = output_;
    }

    void EmbeddingLayer::Backward(const DataSamples& input, const boost::shared_ptr<RowRealMatrix>& gradinput) {
        ParamGrad(input, gradinput);
    }

    void EmbeddingLayer::ParamGrad(const DataSamples& input, const boost::shared_ptr<RowRealMatrix>& gradinput) {
        for (int i = 0; i < input.rows(); ++i) {
            for (DataSamples::InnerIterator iter(input, i); iter; ++iter) {
                grad_.row(iter.col()) += gradinput->col(i);
            }
        }
    }
}
